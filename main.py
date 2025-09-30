import asyncio
import os
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Union
from tqdm.asyncio import tqdm

from experiment.dataset import load_data
from experiment.utils import (
    duration_formatter,
    load_json,
    save_json,
    get_next_log_file,
    get_file_count,
    cost_estimate,
)
from experiment.module_atomic import set_module
from experiment.module_atomic import global_usage_logger
from experiment.module_utils import METHOD_HYPERPARAMS, METHOD_CONFIGS

# Configuration constants
LOG_DIR = "log/{model}/{dataset}/{size}"


# Dataset configuration
@dataclass
class DatasetConfig:
    question_key: Union[str, List[str]]
    answer_key: str
    module_type: str
    scoring_function: str

    def requires_context(self) -> bool:
        return self.module_type == "multi-hop"


# Dataset configuration mapping
DATASET_CONFIGS = {
    "math_level5":
    DatasetConfig(question_key="problem",
                  answer_key="solution",
                  module_type="math",
                  scoring_function="score_math"),
    "aime24":
    DatasetConfig(question_key="Problem",
                  answer_key="Answer",
                  module_type="math",
                  scoring_function="score_math"),
    "aime25":
    DatasetConfig(question_key="problem",
                  answer_key="answer",
                  module_type="math",
                  scoring_function="score_math"),
    "zebra_puzzles":
    DatasetConfig(question_key="question",
                  answer_key="answer",
                  module_type="logical-zebra",
                  scoring_function="exact_match_score"),
    "mini_sudoku":
    DatasetConfig(question_key="question",
                  answer_key="metadata",
                  module_type="logical-sudoku",
                  scoring_function="sudoku_match_score"),
}


class ExperimentRunner:

    def __init__(self,
                 dataset: str,
                 model: str,
                 cot_path: str = None,
                 start: int = 0,
                 end: int = -1,
                 method: str = "dag-sc",
                 # max-number of concurrent tasks, to not exceeding the rate limit of the organization
                 max_concurrent_tasks: int = 30,
                 method_kwargs: Dict[str, Any] = {}):
        # Initialize experiment runner
        self.dataset = dataset
        self.start = start
        self.end = None if end == -1 else end
        self.interval = "full" if self.end is None else f"{start}-{end}"
        self.timestamp = time.time()
        self.method = method
        self.method_kwargs = method_kwargs
        self.max_concurrent_tasks = max_concurrent_tasks
        self.model = model
        self.cot_path = cot_path
        # Validate dataset support
        if dataset not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset}")

        self.config = DATASET_CONFIGS[dataset]

    async def gather_results(self, testset: List[Dict[str, Any]]) -> List[Any]:
        # Collect experiment results
        set_module(self.config.module_type)
        question_key = self.config.question_key

        reason_method = METHOD_CONFIGS[self.method]
        global LOG_DIR
        if self.method not in LOG_DIR:
            LOG_DIR = os.path.join(os.path.dirname(
                LOG_DIR), self.method, os.path.basename(LOG_DIR))

        if isinstance(question_key, list):
            tasks = [
                reason_method(
                    model=self.model,
                    question=self._format_question_from_keys(
                        item, question_key),
                    cot_response=item["cot_response"],
                    **self.method_kwargs
                )
                for item in testset
            ]
        else:
            tasks = [reason_method(model=self.model,
                                   question=item[question_key],
                                   cot_response=item["cot_response"],
                                   **self.method_kwargs) for item in testset]

        # Limit concurrent tasks to avoid overwhelming the system
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        async def limited_task(task):
            async with semaphore:
                return await task

        limited_tasks = [limited_task(task) for task in tasks]
        return await tqdm.gather(*limited_tasks,
                                 desc=f"Processing {self.dataset} tasks")

    def _format_question_from_keys(self, item: Dict[str, Any],
                                   keys: List[str]) -> str:
        # When question_key is a list, concatenate values from multiple keys into a single question
        parts = []
        for key in keys:
            if key in item:
                parts.append(f"{key}: {item[key]}")
        return "\n".join(parts)

    def construct_entry(self, result: Tuple[Dict[str, Any], Any],
                        data: Dict[str, Any]) -> Dict[str, Any]:
        # Construct result entry
        result_data, log = result
        question_key = self.config.question_key
        answer_key = self.config.answer_key

        # Handle case where question_key is a list
        if isinstance(question_key, list):
            question = self._format_question_from_keys(data, question_key)
        else:
            question = data[question_key]

        groundtruth = data[answer_key]

        entry = {
            "problem": question,
            "groundtruth": groundtruth,
            "response": result_data.get("response"),
            "answer": result_data.get("answer"),
            "log": log
        }

        # Dynamically import scoring function
        scoring_function = getattr(
            __import__(f"experiment.utils",
                       fromlist=[self.config.scoring_function]),
            self.config.scoring_function)

        # Pass different parameters based on scoring function
        if self.config.scoring_function == "score_math":
            entry["score"] = scoring_function(entry["answer"], groundtruth,
                                              self.dataset)
        else:
            entry["score"] = scoring_function(entry["answer"], groundtruth)
        return entry

    async def update_score_log(self, accuracy: float) -> None:
        # Update score log
        call_count, total_prompt_tokens, total_completion_tokens = await global_usage_logger.get_total()
        log_entry = {
            "start": self.start,
            "end": self.end,
            "token": {
                "prompt": total_prompt_tokens,
                "completion": total_completion_tokens
            },
            "call_count": call_count,
            "accuracy": accuracy,
            "cost": cost_estimate(model=self.model,
                                  num_input_tokens=total_prompt_tokens,
                                  num_output_tokens=total_completion_tokens)
        }

        score_log_file = LOG_DIR.format(model=self.model,
                                        dataset=self.dataset,
                                        size=self.interval) + "/score.json"
        existing_log = load_json(score_log_file) if os.path.exists(
            score_log_file) else {}
        count = get_file_count(LOG_DIR,
                               self.model,
                               self.interval,
                               self.dataset,
                               exclude_score=True)

        if self.dataset not in existing_log:
            existing_log[self.dataset] = {}
        existing_log[self.dataset][str(count)] = log_entry
        save_json(score_log_file, existing_log)

    async def run(self) -> float:
        # Run experiment and return accuracy
        print(
            f"Running {self.method} experiment on {self.dataset} dataset from index {self.start} to {self.end}"
        )

        # Load test set (only the subset we need)
        # if cot_path is not None, we will use the original CoT responses, to make fair comparison
        testset = load_data(self.dataset, "test", self.cot_path)[
            self.start:self.end]

        # minibatch for the experiment run, in case of the error,
        # we can save the partial results and continue the experiment.
        minibatch_size = 60
        results = []
        for i in range(0, len(testset), minibatch_size):
            testset_batch = testset[i:i + minibatch_size]
            results_batch = await self.gather_results(testset_batch)

            results.extend(results_batch)

            # Build results
            json_obj = [
                self.construct_entry(result, data)
                for result, data in zip(results, testset[:len(results)])
            ]
            accuracy = sum(entry["score"]
                           for entry in json_obj) / len(json_obj)

            # Save results
            interval = f"{self.start}-{self.start + len(results)}"
            log_file = get_next_log_file(
                LOG_DIR, self.model, interval, self.dataset)
            save_json(log_file, json_obj)

            # Update score log
            await self.update_score_log(accuracy)

            # Print result summary
            print(f"Unsolved: {round((1-accuracy) * len(json_obj))}")
            print(f"Accuracy: {accuracy:.4f}")
            print(
                f"Time taken: {duration_formatter(time.time() - self.timestamp)}"
            )

        return accuracy


async def main():
    # Main function
    parser = argparse.ArgumentParser(
        description='Run experiments on various datasets')
    parser.add_argument(
        '--method',
        type=str,
        choices=METHOD_CONFIGS.keys(),
        default='dag-sc',
        help='Method: dag-sc (DAG with self-consistency), dag (DAG), sc (self-consistency), atom (standard experiment), tot (top-down), fot (bottom-up)'
    )

    # Parse known args first to avoid unrecognized arguments error
    args, _ = parser.parse_known_args()

    # Add dataset-specific arguments to parser
    parser.add_argument('--dataset',
                        type=str,
                        default='math',
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset to run experiment on')
    parser.add_argument('--cot-path',
                        type=str,
                        default=None,
                        help='Path to the CoT responses')
    parser.add_argument('--start',
                        type=int,
                        default=0,
                        help='Start index of the dataset')
    parser.add_argument('--end',
                        type=int,
                        default=-1,
                        help='End index of the dataset (-1 for all)')
    parser.add_argument('--model',
                        type=str,
                        default='gpt-4o-mini',
                        help='Model to use for the experiment')
    parser.add_argument('--max-concurrent-tasks',
                        type=int,
                        default=10,
                        help='Max concurrent tasks for the experiment')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Verbose output')
    parser.add_argument('--temperature',
                        type=float,
                        default=0.6,
                        help='Temperature for the model')
    parser.add_argument('--max-tokens',
                        type=int,
                        default=8192,
                        help='Max tokens for the model')

    # Add method-specific arguments to parser
    method = args.method if hasattr(args, "method") else "cot"
    for m, params in METHOD_HYPERPARAMS.items():
        if m == method:
            for name, typ, default, help_str in params:
                # for store_true, we need to add the argument with action="store_true"
                if typ == "store_true":
                    parser.add_argument(
                        f"--{name.replace('_', '-')}", action="store_true", help=f"[{m}] {help_str}")
                    continue
                # Avoid adding duplicate arguments
                arg_name = name.replace('_', '-')
                if any(a.lstrip('-').replace('-', '_') == name for a in parser._option_string_actions):
                    continue
                arg_name = f"--{name.replace('_', '-')}"
                parser.add_argument(arg_name, type=typ,
                                    default=default, help=f"[{m}] {help_str}")

    # Parse again to get method-specific args
    args = parser.parse_args()
    print("--------------------------------")
    print("General Args:", args)
    print("--------------------------------")

    # Collect method_kwargs for ExperimentRunner
    method_kwargs = {}

    # For shared arguments between methods (e.g., verbose, temperature, etc.)
    method_kwargs["verbose"] = args.verbose
    method_kwargs["temperature"] = args.temperature
    method_kwargs["max_tokens"] = args.max_tokens

    for name, typ, default, _ in METHOD_HYPERPARAMS.get(args.method, []):
        value = getattr(args, name, default)
        method_kwargs[name] = value

    if method == "mctsr":
        method_kwargs["dataset"] = args.dataset
    elif method == "cot":
        method_kwargs["num_samples"] = 1

    args.method_kwargs = method_kwargs

    print("--------------------------------")
    print("Method Args:", method_kwargs)
    print("--------------------------------")

    # Run standard experiment
    runner = ExperimentRunner(dataset=args.dataset,
                              model=args.model,
                              cot_path=args.cot_path,
                              start=args.start,
                              end=args.end,
                              method=args.method,
                              max_concurrent_tasks=args.max_concurrent_tasks,
                              method_kwargs=args.method_kwargs)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
