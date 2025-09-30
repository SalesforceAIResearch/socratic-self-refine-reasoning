import google.auth.transport.requests
from google.auth import default
import json
import numpy as np

from experiment.utils import (
    extract_json,
    score_math,
    score_mc,
    score_mh,
    parse_answer,
    majority_vote_answers,
    sudoku_match_score,
    cryptarithm_match_score_iou,
    score_zebra,
)

from experiment.prompter import math, logical
import asyncio
import copy
import random
from collections import Counter

from apikey import url, openai_api_key
import openai
from google import genai


count = 0
MAX_RETRIES = 5
LABEL_RETRIES = 3
ATOM_DEPTH = 3
# use asyncio semaphore to limit the number of concurrent requests
MAX_CONCURRENT_ATOMIC_REQUESTS = 100
semaphore = asyncio.Semaphore(MAX_CONCURRENT_ATOMIC_REQUESTS)

score = None
module = None
prompter = None


def set_module(module_name):
    global module, prompter, score
    module = module_name
    if module == "math":
        prompter = math
        score = score_math
    elif module == "logical-zebra":
        prompter = logical
        score = score_zebra
    elif module == "logical-sudoku":
        prompter = logical
        score = sudoku_match_score


def with_semaphore(func):
    """Decorator to run an async function with the global semaphore."""
    async def wrapper(*args, **kwargs):
        async with semaphore:
            # print(
            #     f"Available Semaphre:{semaphore._value}/{MAX_CONCURRENT_ATOMIC_REQUESTS}")
            return await func(*args, **kwargs)
    return wrapper


openai_client = openai.AsyncClient(api_key=openai_api_key[0], base_url=url)


# Google Gemini Client

# TODO(developer): Update and un-comment below lines
project_id = "<your-project-id>"
location = "<your-location>"

# # Programmatically get an access token
credentials, _ = default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"])
credentials.refresh(google.auth.transport.requests.Request())

# OpenAI Client
google_client = openai.AsyncClient(
    base_url=f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi",
    api_key=credentials.token
)


class DummyMessage:
    def __init__(self, content="Sorry, I don't know the answer."):
        self.content = content


class DummyChoice:
    def __init__(self, content="Sorry, I don't know the answer."):
        self.message = DummyMessage(content=content)


class DummyResponse:
    def __init__(self, content="Sorry, I don't know the answer."):
        self.choices = [DummyChoice(content=content)]
        self.usage = None


def retry(max_retries=100):
    """Retry decorator for async functions in DAG."""

    def decorator(func):

        async def wrapper(*args, **kwargs):
            retries = max_retries
            while retries >= 0:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    print(f"Error log: {e}")
                    if retries == 0 or "error code: 422" in str(e).lower():
                        print(f"Max retries reached and returning dummy response ")
                        return DummyResponse()
                    retries -= 1
                    # Exponential backoff
                    await asyncio.sleep(random.uniform(0.1, 1))
            return None

        return wrapper

    return decorator


class GlobalUsageLogger:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0
        self.lock = asyncio.Lock()

    async def add_tokens(self, prompt_tokens, completion_tokens):
        async with self.lock:
            self.total_prompt_tokens += (
                prompt_tokens if prompt_tokens is not None else 0)
            self.total_completion_tokens += (
                completion_tokens if completion_tokens is not None else 0)

    async def get_total(self):
        async with self.lock:
            return self.call_count, self.total_prompt_tokens, self.total_completion_tokens

    async def add_calls(self, num_calls: int = 1):
        async with self.lock:
            self.call_count += (num_calls if num_calls is not None else 0)


global_usage_logger = GlobalUsageLogger()


def usage_logger(func):
    """Decorator to log and accumulate token usage for LLM calls."""
    async def wrapper(*args, **kwargs):
        response = await func(*args, **kwargs)
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = 0
            completion_tokens = 0
            # Try to get total tokens from OpenAI or Together response
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
            elif hasattr(usage, "prompt_tokens"):
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
            await global_usage_logger.add_tokens(prompt_tokens, completion_tokens)
            await global_usage_logger.add_calls(1)
            call_count, total_prompt_tokens, total_completion_tokens = await global_usage_logger.get_total()
            if call_count % 100 == 0:
                print(f"#Calls: {call_count}")
                print(
                    f"Total token usage: {total_prompt_tokens} prompt tokens, {total_completion_tokens} completion tokens.")
        else:
            print("Token usage info not available in response.")
        return response
    return wrapper


@usage_logger
@retry()
@with_semaphore
async def call_llm(*args, **kwargs):
    """Call the LLM and update the token usage."""
    openai_models = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-5-nano",
        "gpt-5-mini",
    ]
    google_models = [
        "google/gemini-2.5-flash-lite",
        "google/gemini-2.5-flash",
    ]

    model = kwargs["model"]

    # print(f"Calling {model} with {kwargs}")
    if model in openai_models:
        client = openai_client
    elif model in google_models:
        client = google_client
    else:
        print(f"Model {model} not supported")
        raise ValueError(f"Model {model} not supported")

    if model == "gpt-5-nano" or model == "gpt-5-mini":
        if "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs["max_tokens"]
            del kwargs["max_tokens"]
        if "temperature" in kwargs:
            del kwargs["temperature"]
        kwargs["reasoning_effort"] = "minimal"
        kwargs["verbosity"] = "medium"

    # Functionally, there is no difference between using the response api and the chat api.
    response = await client.chat.completions.create(*args, **kwargs)
    return response


async def call_llm_with_history(prompt, history=[], truncate=True, model="gpt-4o-mini", verbose=False, **kwargs):
    """LLM call with history. For MCTSr reasoning."""
    history_ = [{"role": "user" if i % 2 == 0 else 'assistant',
                 "content": h} for i, h in enumerate(history)]
    if truncate:
        history_ = history_[-2:]

    messages = history_ + [{"role": "user", "content": prompt}]

    response = await call_llm(model=model, messages=messages, **kwargs)
    content = response.choices[0].message.content

    # here the scores are not used in the main code
    return content, list(history) + [prompt, content], 0


async def direct(question: str,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.6,
                 num_samples: int = 1,
                 max_tokens: int = 8192,
                 verbose: bool = False,
                 **kwargs) -> list[str]:
    """[CoT] reasoning."""
    prompt = getattr(prompter, "direct")(model=model, question=question)
    response = await call_llm(model=model,
                              messages=[{"role": "user", "content": prompt}],
                              temperature=temperature,
                              max_tokens=max_tokens,
                              n=num_samples)
    if verbose:
        print("-" * 100)
        print(f"Direct response for {question}:")
        print(response.choices[0].message.content)
        print("-" * 100)

    return [choice.message.content for choice in response.choices]


async def multistep(question: str,
                    model: str = "gpt-4o-mini",
                    temperature: float = 0.6,
                    num_samples: int = 5,
                    max_tokens: int = 8192,
                    verbose: bool = False,
                    **kwargs) -> list[str]:
    prompt = getattr(prompter, "multistep")(model=model, question=question)
    response = await call_llm(model=model,
                              messages=[{"role": "user", "content": prompt}],
                              temperature=temperature,
                              max_tokens=max_tokens,
                              n=num_samples)
    if verbose:
        print("-" * 100)
        print(f"Multistep response for {question}:")
        print(response.choices[0].message.content)
        print("-" * 100)

    return [choice.message.content for choice in response.choices]


################################################################################
# Atom of Thought (AOT)
################################################################################

async def decompose_w_dependencies(question: str,
                                   trajectory: str,
                                   answer: str,
                                   model: str = "gpt-4o-mini",
                                   temperature: float = 0.6,
                                   max_tokens: int = 8192,
                                   verbose: bool = False,
                                   **kwargs) -> dict:
    prompt = getattr(prompter, "decompose_w_dependencies")(model=model,
                                                           question=question,
                                                           trajectory=trajectory,
                                                           answer=answer)
    response = await call_llm(model=model,
                              messages=[{"role": "user", "content": prompt}],
                              temperature=temperature,
                              max_tokens=max_tokens)

    json_obj = extract_json(response.choices[0].message.content)

    # added one-step of retry.
    if isinstance(json_obj, str):
        error_message = json_obj
        original_response = response.choices[0].message.content
        response = await call_llm(model=model,
                                  messages=[
                                      {"role": "user", "content": prompt},
                                      {"role": "assistant",
                                          "content": original_response},
                                      {"role": "user", "content": f"The original response is not a valid JSON object. Please fix it based on the error message: {error_message}"}
                                  ],
                                  temperature=temperature,
                                  max_tokens=max_tokens)
        json_obj = extract_json(response.choices[0].message.content)

    if isinstance(json_obj, str) or "sub-questions" not in json_obj:
        sub_qa_pairs = []
    else:
        sub_qa_pairs = json_obj["sub-questions"]

    if verbose:
        print("-" * 100)
        print(f"Decompose with dependencies response for\n{question}:\n")
        print(response.choices[0].message.content)
        print("-" * 100)
        print("JSON object:")
        print(json.dumps(json_obj, indent=4))
        print("Sub-questions:")
        print(sub_qa_pairs)
        print("-" * 100)

    return sub_qa_pairs


async def contract(question: str,
                   decompose_result: dict,
                   independent: list[dict] = None,
                   dependent: list[dict] = None,
                   model: str = "gpt-4o-mini",
                   temperature: float = 0.6,
                   max_tokens: int = 8192,
                   is_socratic: bool = False,
                   verbose: bool = False,
                   **kwargs) -> str:

    prompt_func = getattr(prompter, "contract") if not is_socratic else getattr(
        prompter, "socratic_contract")
    prompt = prompt_func(model=model,
                         question=question,
                         decompose_result=decompose_result,
                         independent=independent,
                         dependent=dependent)
    response = await call_llm(model=model,
                              messages=[{"role": "user", "content": prompt}],
                              temperature=temperature,
                              max_tokens=max_tokens)
    ret = response.choices[0].message.content
    if verbose:
        print("-" * 100)
        print(f"Contract response for {question}:")
        print(ret)
        print("-" * 100)

    return ret


async def ensemble(question: str,
                   results: list[str],
                   model: str = "gpt-4o-mini",
                   temperature: float = 0.6,
                   max_tokens: int = 8192,
                   verbose: bool = False,
                   **kwargs) -> str:
    prompt = getattr(prompter, "ensemble")(model=model,
                                           question=question,
                                           solutions=results)
    response = await call_llm(model=model,
                              messages=[{"role": "user", "content": prompt}],
                              temperature=temperature,
                              max_tokens=max_tokens)
    ret = response.choices[0].message.content
    if verbose:
        print("-" * 100)
        print(f"Ensemble response for {question}:")
        print(ret)
        print("-" * 100)

    return ret


################################################################################
# Self-Refine
################################################################################

async def judge(question: str,
                trajectory: str,
                model: str = "gpt-4o-mini",
                temperature: float = 0.6,
                max_tokens: int = 8192,
                verbose: bool = False,
                **kwargs) -> str:
    messages = getattr(prompter, "judge")(model=model,
                                          question=question,
                                          trajectory=trajectory)
    response = await call_llm(model=model,
                              messages=messages,
                              temperature=temperature,
                              max_tokens=max_tokens,
                              **kwargs)
    ret = response.choices[0].message.content
    if verbose:
        print("-" * 100)
        print(f"Judge response for {question}:")
        print(ret)
        print("-" * 100)

    return ret


async def refine(question: str,
                 original_cot_response: str,
                 judge_response: str,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.6,
                 max_tokens: int = 8192,
                 verbose: bool = False,
                 **kwargs) -> str:
    prompt = getattr(prompter, "refine")(model=model,
                                         question=question,
                                         original_cot_response=original_cot_response,
                                         judge_response=judge_response)
    response = await call_llm(model=model,
                              messages=[{"role": "user", "content": prompt}],
                              temperature=temperature,
                              max_tokens=max_tokens,
                              **kwargs)
    ret = response.choices[0].message.content
    if verbose:
        print("-" * 100)
        print(f"Refine response for {question}:")
        print(ret)
        print("-" * 100)

    return ret


################################################################################
# LLM-Debate (Debate)
################################################################################

async def debate_refine(question: str,
                        original_cot_response: str,
                        other_agents_responses: list[str],
                        model: str = "gpt-4o-mini",
                        temperature: float = 0.6,
                        max_tokens: int = 8192,
                        verbose: bool = False,
                        **kwargs) -> str:
    prompt = getattr(prompter, "debate_refine")(model=model,
                                                question=question,
                                                original_cot_response=original_cot_response,
                                                other_agents_responses=other_agents_responses)
    response = await call_llm(model=model,
                              messages=[{"role": "user", "content": prompt}],
                              temperature=temperature,
                              max_tokens=max_tokens,
                              **kwargs)
    ret = response.choices[0].message.content
    if verbose:
        print("-" * 100)
        print(f"Refine response for {question}:")
        print(ret)
        print("-" * 100)

    return ret


################################################################################
# Socratic Self-Refine
################################################################################

async def decompose(question: str,
                    cot_response: str,
                    model: str = "gpt-4o-mini",
                    temperature: float = 0.0,
                    max_tokens: int = 8192,
                    verbose: bool = False):
    """Decompose the reasoning process into a series of (sub-q, sub-a) pairs."""
    prompt = getattr(prompter, "decompose")(model=model,
                                            question=question,
                                            cot_response=cot_response)
    response = await call_llm(model=model,
                              messages=[{"role": "user", "content": prompt}],
                              temperature=temperature,
                              max_tokens=max_tokens,
                              response_format={"type": "json_object"})

    json_obj = extract_json(response.choices[0].message.content)
    # added one-step of retry.
    if isinstance(json_obj, str):
        error_message = json_obj
        original_response = response.choices[0].message.content
        response = await call_llm(model=model,
                                  messages=[{"role": "user", "content": prompt},
                                            {"role": "assistant",
                                            "content": original_response},
                                            {"role": "user", "content": f"The original response is not a valid JSON object. Please fix it based on the error message: {error_message}"}],
                                  temperature=temperature,
                                  max_tokens=max_tokens)
        json_obj = extract_json(response.choices[0].message.content)
    if "llama" in model.lower():
        try:
            sub_qa_pairs = [{"description": json_obj["sub-questions"][i],
                            "answer": json_obj["sub-answers"][i]} for i in range(len(json_obj["sub-questions"]))]
        except:
            sub_qa_pairs = []
    else:
        try:
            sub_qa_pairs = json_obj["sub-questions"]
        except:
            sub_qa_pairs = []

    if verbose:
        print("-" * 100)
        print(f"Decompose response for question: \n\n{question}\n\n")
        print(response.choices[0].message.content)
        print("-" * 100)
        print("Sub-questions:")
        print(sub_qa_pairs)
        print("-" * 100)

    return sub_qa_pairs


async def decompose_w_max_steps(question: str,
                                cot_response: str,
                                model: str = "gpt-4o-mini",
                                temperature: float = 0.0,
                                max_tokens: int = 8192,
                                max_steps: int = 3,
                                verbose: bool = False):
    """Decompose the reasoning process into a series of (sub-q, sub-a) pairs."""
    prompt = getattr(prompter, "decompose_w_max_steps")(model=model,
                                                        question=question,
                                                        cot_response=cot_response,
                                                        max_steps=max_steps)
    response = await call_llm(model=model,
                              messages=[{"role": "user", "content": prompt}],
                              temperature=temperature,
                              max_tokens=max_tokens,
                              response_format={"type": "json_object"})

    json_obj = extract_json(response.choices[0].message.content)
    # added one-step of retry.
    if isinstance(json_obj, str):
        error_message = json_obj
        original_response = response.choices[0].message.content
        response = await call_llm(model=model,
                                  messages=[{"role": "user", "content": prompt},
                                            {"role": "assistant",
                                            "content": original_response},
                                            {"role": "user", "content": f"The original response is not a valid JSON object. Please fix it based on the error message: {error_message}"}],
                                  temperature=temperature,
                                  max_tokens=max_tokens)
        json_obj = extract_json(response.choices[0].message.content)
    if "llama" in model.lower():
        try:
            sub_qa_pairs = [{"description": json_obj["sub-questions"][i],
                            "answer": json_obj["sub-answers"][i]} for i in range(len(json_obj["sub-questions"]))]
        except:
            sub_qa_pairs = []
    else:
        try:
            sub_qa_pairs = json_obj["sub-questions"]
        except:
            sub_qa_pairs = []

    if verbose:
        print("-" * 100)
        print(f"Decompose response for question: \n\n{question}\n\n")
        print(response.choices[0].message.content)
        print("-" * 100)
        print("Sub-questions:")
        print(sub_qa_pairs)
        print("-" * 100)

    return sub_qa_pairs


async def check_equivalence(prediction,
                            reference_answers,
                            model: str = "gpt-4o-mini",
                            verbose: bool = False):
    """For equivalence checking, we use the cheap model."""
    prompt = getattr(prompter, "check_equivalence")(model=model,
                                                    prediction=prediction,
                                                    reference_answers=reference_answers)

    response = await call_llm(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    answer = response.choices[0].message.content
    if verbose:
        print("-" * 100)
        print(
            f"Check equivalence. Prediction: {prediction}.\n Reference answers: {reference_answers}")
        print(answer)
        print("-" * 100)
    try:
        final_answer = int(answer.split("<answer>")[
                           1].split("</answer>")[0].strip())
    except Exception:
        final_answer = -1
    return final_answer


async def confidence_estimate(question: str,
                              sub_question_answer_pairs: list,
                              start_index: int = 0,
                              model: str = "gpt-4o-mini",
                              temperature: float = 0.6,
                              max_tokens: int = 8192,
                              context_type: str = "naive",
                              verbose: bool = False,
                              **kwargs):

    def get_preceding_sub_questions(sub_question_answer_pairs: list, index: int, naive: bool = True):
        if len(sub_question_answer_pairs) == 0:
            return [], []

        if naive:
            return sub_question_answer_pairs[:index]

        assert all("depend" in sub_question_answer_pairs[i].keys() for i in range(
            index)), "All sub-questions before the current index must have dependencies."

        # For the index-th node, find all necessary nodes (direct and indirect dependencies)
        def get_all_dependencies(sub_question_answer_pairs: list, index: int) -> list:
            """Return a topologically sorted list of all indices that the index-th node depends on (directly or indirectly)."""
            # First, collect all dependencies (direct and indirect) using DFS
            visited = set()

            def dfs(node_idx):
                if node_idx >= len(sub_question_answer_pairs) or node_idx < 0:
                    return
                for dep in sub_question_answer_pairs[node_idx].get("depend", []):
                    if dep not in visited:
                        visited.add(dep)
                        dfs(dep)
            dfs(index)
            # Now, perform topological sort on the visited nodes

            def topo_sort(nodes, sub_question_answer_pairs):
                in_degree = {i: 0 for i in nodes}
                graph = {i: [] for i in nodes}
                for i in nodes:
                    if i < 0 or i >= len(sub_question_answer_pairs):
                        continue  # skip invalid dependency indices
                    for dep in sub_question_answer_pairs[i].get("depend", []):
                        if dep in nodes:
                            graph[dep].append(i)
                            in_degree[i] += 1
                queue = [n for n in nodes if in_degree[n] == 0]
                result = []
                while queue:
                    n = queue.pop(0)
                    result.append(n)
                    for m in graph[n]:
                        in_degree[m] -= 1
                        if in_degree[m] == 0:
                            queue.append(m)
                return result
            return topo_sort(visited, sub_question_answer_pairs)

        preceding_indices = get_all_dependencies(
            sub_question_answer_pairs, index)

        return [sub_question_answer_pairs[i] for i in preceding_indices if i >= 0 and i < len(sub_question_answer_pairs)]

    batch_messages = [
        getattr(prompter,
                "socratic_answer_next_sub_question")(model=model,
                                                     question=question,
                                                     sub_question_answer_pairs=get_preceding_sub_questions(
                                                         sub_question_answer_pairs, i, naive=context_type == "naive"),
                                                     next_sub_question=sub_question_answer_pairs[i]["description"])
        for i in range(start_index, len(sub_question_answer_pairs))
    ]

    if verbose:
        print("-" * 100)
        print("Confidence estimate batch messages:")
        print(batch_messages)
        print("-" * 100)

    tasks = [
        call_llm(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=5,
        )
        for messages in batch_messages
    ]

    results = await asyncio.gather(*tasks)
    if verbose:
        print("-" * 100)
        print("Confidence estimate results:")
        print(results)
        print("-" * 100)

    step_answers = [[parse_answer(choice.message.content)
                     for choice in result.choices] for result in results]
    if verbose:
        print("-" * 100)
        print("Step answers:")
        print(step_answers)
        print("-" * 100)

    answer_equiv_tasks = [check_equivalence(
        a["answer"], step_answer, model=model) for a, step_answer in zip(sub_question_answer_pairs, step_answers)]
    answer_equiv_scores = await asyncio.gather(*answer_equiv_tasks)

    if verbose:
        print("-" * 100)
        print("Answer equivalence scores:")
        print(answer_equiv_scores)
        print("-" * 100)

    # estimate the confidence of the answers:
    return step_answers, answer_equiv_scores


async def decompose_and_confidence_estimate(question: str,
                                            cot_response: str,
                                            model: str = "gpt-4o-mini",
                                            temperature: float = 0.0,
                                            max_tokens: int = 8192,
                                            verbose: bool = False,
                                            **kwargs):
    try:
        sub_qa_pairs = await decompose(question=question,
                                       cot_response=cot_response,
                                       model=model,
                                       temperature=0.0,
                                       max_tokens=max_tokens,
                                       verbose=verbose)
        socratic_judge_results = await confidence_estimate(question=question,
                                                           sub_question_answer_pairs=sub_qa_pairs,
                                                           model=model,
                                                           temperature=temperature,
                                                           max_tokens=max_tokens,
                                                           verbose=verbose)
        ret = sub_qa_pairs, socratic_judge_results
    except Exception:
        ret = [], [[], []]
    return ret


async def socratic_refine(direct_instruction: str,
                          cot_trajectory: str,
                          socratic_questions: list[str],
                          socratic_answers: list[str],
                          socratic_answer_supports: list[str],
                          socratic_confidence_scores: list[int],
                          model: str = "gpt-4o-mini",
                          temperature: float = 0.6,
                          max_tokens: int = 8192,
                          verbose: bool = False,
                          **kwargs) -> str:

    min_confidence_step = np.argmin(socratic_confidence_scores)
    wrong_question = socratic_questions[min_confidence_step]
    wrong_answer = socratic_answers[min_confidence_step]
    revised_answer = majority_vote_answers(
        socratic_answer_supports[min_confidence_step])

    reflection = getattr(prompter, "socratic_reflection")(model=model,
                                                          wrong_question=wrong_question,
                                                          wrong_answer=wrong_answer,
                                                          revised_answer=revised_answer)

    refine_messages = getattr(prompter, "socratic_refine")(model=model,
                                                           direct_instruction=direct_instruction,
                                                           cot_trajectory=cot_trajectory,
                                                           reflection=reflection)

    if verbose:
        print("-" * 100)
        print("Socratic judge reflection:")
        print(reflection)
        print("-" * 100)
        print("Socratic judge refine messages:")
        print(refine_messages)
        print("-" * 100)

    response = await call_llm(model=model,
                              messages=refine_messages,
                              temperature=temperature,
                              max_tokens=max_tokens,
                              **kwargs)

    return response.choices[0].message.content


################################################################################
# Socratic Self-Refine w/ Plan Refinement (SSR-Plan)
################################################################################

async def plan_summarize(question: str,
                         cot_response: str,
                         model: str = "gpt-4o-mini",
                         temperature: float = 0.6,
                         max_tokens: int = 8192,
                         verbose: bool = False,
                         **kwargs) -> str:
    prompt = getattr(prompter, "plan_summarize")(model=model,
                                                 question=question,
                                                 cot_response=cot_response)
    resposne = await call_llm(model=model,
                              messages=[{"role": "user", "content": prompt}],
                              temperature=temperature,
                              max_tokens=max_tokens,
                              **kwargs)
    if verbose:
        print("-" * 100)
        print(f"Plan summarize response for {question}:")
        print(resposne.choices[0].message.content)
        print("-" * 100)
    return resposne.choices[0].message.content


async def refine_plan(question: str,
                      plan: str,
                      evaluation: str,
                      model: str = "gpt-4o-mini",
                      temperature: float = 0.6,
                      max_tokens: int = 8192,
                      verbose: bool = False,
                      **kwargs) -> str:
    prompt = getattr(prompter, "refine_plan")(model=model,
                                              question=question,
                                              evaluation=evaluation,
                                              plan=plan)
    response = await call_llm(model=model,
                              messages=[{"role": "user", "content": prompt}],
                              temperature=temperature,
                              max_tokens=max_tokens,
                              **kwargs)
    return response.choices[0].message.content


async def direct_w_plan(question: str,
                        plan: str,
                        model: str = "gpt-4o-mini",
                        temperature: float = 0.6,
                        max_tokens: int = 8192,
                        verbose: bool = False,
                        **kwargs) -> str:
    prompt = getattr(prompter, "direct_w_plan")(model=model,
                                                question=question,
                                                plan=plan)
    response = await call_llm(model=model,
                              messages=[{"role": "user", "content": prompt}],
                              temperature=temperature,
                              max_tokens=max_tokens,
                              **kwargs)
    if verbose:
        print("-" * 100)
        print(f"CoT with plan response for {question}:")
        print(response.choices[0].message.content)
        print("-" * 100)
    return response.choices[0].message.content
