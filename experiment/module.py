
from experiment.utils import parse_answer
import asyncio
import copy
import random
from collections import Counter

from experiment.module_atomic import (
    direct,
    multistep,
    decompose_w_dependencies,
    contract,
    ensemble,
    judge,
    refine,
    debate_refine,
)

count = 0
MAX_RETRIES = 5
DECOMPOSE_RETRIES = 3
ATOM_DEPTH = 3


async def self_consistency(model: str,
                           question: str,
                           # dummy argument for compatibility
                           cot_response: str = None,
                           num_samples: int = 5,
                           log=None,
                           **kwargs):
    """
    This function is used to reason about the question in a self-consistency way.
    It first decomposes the question into a DAG, then it use topological sort to get the order of the nodes.
    Then it execute the nodes in order (each step in parallel for step-level self-consistency).
    Finally, it summarizes the results to get the final answer.
    """
    # Initialize logging
    log = log if log else {}
    index = len(log)
    log[index] = {}

    # if n=1, the returned is not a list, but a string
    self_consistency_responses = await direct(model=model, question=question, num_samples=num_samples, **kwargs)
    self_consistency_answers = [parse_answer(response) for response in self_consistency_responses
                                ]  # extract "answer" from the response.
    self_consistency_results = [{
        "response": response,
        "answer": answer
    } for response, answer in zip(self_consistency_responses, self_consistency_answers)]

    # filter out the NA answers
    non_na_results = [result for result in self_consistency_results
                      if result["answer"] != "NA"]
    if len(non_na_results) > 0:
        answer_counter = Counter([result["answer"]
                                  for result in non_na_results])
        most_common_answer = answer_counter.most_common(1)[0][0]
    else:
        most_common_answer = "NA"

    log[index].update({
        "self_consistency_results": self_consistency_results,
        "answers": self_consistency_answers,
        "most_common_answer": most_common_answer,
    })
    return {"response": most_common_answer, "answer": most_common_answer}, log


async def self_refine(model: str,
                      question: str,
                      cot_response: str = None,
                      max_iter: int = 1,
                      log=None,
                      **kwargs):
    """
    This function is used to reason about the question using the self-refine algorithm.
    It first decomposes the question into a DAG, then it use topological sort to get the order of the nodes.
    Then it execute the nodes in order (each step in parallel for step-level self-consistency).
    Finally, it summarizes the results to get the final answer.
    """
    # Initialize logging
    log = log if log else {}
    index = len(log)
    log[index] = {}

    # Get the initial cot-trajectory
    if cot_response is not None:
        cot_trajectory = cot_response
    else:
        cot_trajectory = (await direct(model=model, question=question, **kwargs))[0]
    cot_result = {
        "response": cot_trajectory,
        "answer": parse_answer(cot_trajectory)
    }

    refine_results = []
    for _ in range(max_iter):
        reflection = await judge(model=model,
                                 question=question,
                                 trajectory=cot_trajectory,
                                 **kwargs)
        refined_trajectory = await refine(model=model,
                                          question=question,
                                          original_cot_response=cot_trajectory,
                                          judge_response=reflection,
                                          **kwargs)
        refine_results.append({
            "reflection": reflection,
            "response": refined_trajectory,
            "answer": parse_answer(refined_trajectory)
        })
        cot_trajectory = refined_trajectory

    log[index].update({
        "cot_result": cot_result,
        "refine_results": refine_results,
    })

    return {"response": cot_trajectory, "answer": parse_answer(cot_trajectory)}, log


async def debate(model: str,
                 question: str,
                 cot_response: str = None,
                 num_agents: int = 2,
                 max_iter: int = 1,
                 log=None,
                 **kwargs):
    log = log if log else {}
    index = len(log)
    log[index] = {}

    if cot_response is None:
        first_round_tasks = [
            direct(model=model, question=question, **kwargs) for _ in range(num_agents)]
        first_round_cot_responses = await asyncio.gather(*first_round_tasks)
        all_round_responses = [[response[0]
                                for response in first_round_cot_responses]]

    else:
        first_round_tasks = [
            direct(model=model, question=question, **kwargs) for _ in range(num_agents-1)]
        first_round_cot_responses = await asyncio.gather(*first_round_tasks)
        first_round_cot_responses = [response[0]
                                     for response in first_round_cot_responses]

        all_round_responses = [[cot_response] + first_round_cot_responses]

    for _ in range(max_iter):
        refine_tasks = [debate_refine(model=model,
                                      question=question,
                                      original_cot_response=current_cot,
                                      other_agents_responses=all_round_responses[-1][:i] +
                                      all_round_responses[-1][i+1:],
                                      **kwargs) for i, current_cot in enumerate(all_round_responses[-1])]
        refine_results = await asyncio.gather(*refine_tasks)
        all_round_responses.append(refine_results)

    # randomly pick one from the last round of the responses
    random_response = random.choice(all_round_responses[-1])
    answer = parse_answer(random_response)

    log[index].update({
        "refine_results": all_round_responses,
    })

    return {"response": random_response, "answer": answer}, log


async def mctsr(model: str,
                question: str,
                dataset: str,
                max_iter: int,
                cot_response: str = None,  # dummy argument for compatibility
                log=None,
                **kwargs):
    """
    This is the function to reason about the question using the FoT algorithm.
    """
    # Initialize logging
    log = log if log else {}
    index = len(log)
    log[index] = {}

    from .mctsr.mctsr import monte_carlo_tree
    response, is_activate, activated_answers_list, activated_answer_scores, answers_list, to_explore, to_explore_reward, ucb_bank = await monte_carlo_tree(
        dataset, question, max_iter, "", model, **kwargs)
    answer = parse_answer(response)
    result = {
        "response": response,
        "answer": answer,
        "activated_answers_list": activated_answers_list,
        "activated_answer_scores": activated_answer_scores,
        "answers_list": answers_list,
        "to_explore": to_explore,
        "to_explore_reward": to_explore_reward,
        "ucb_bank": ucb_bank
    }

    log[index].update(result)

    return result, log


async def decompose(model: str, question: str, trajectory: str = None, **kwargs):
    # if trajectory is not provided, use multistep to get the trajectory.
    if trajectory is None:
        trajectory = (await multistep(model=model, question=question, **kwargs))[0]
    multistep_result = {
        "response": trajectory,
        "answer": parse_answer(trajectory)
    }

    retries = DECOMPOSE_RETRIES
    for _ in range(retries):
        sub_qa_pairs = await decompose_w_dependencies(model=model,
                                                      question=question,
                                                      trajectory=multistep_result["response"],
                                                      answer=multistep_result["answer"],
                                                      **kwargs)
        result = {
            "answer": multistep_result["answer"],
            "response": multistep_result["response"],
            "sub-questions": sub_qa_pairs
        }

        # Here we don't use the original AoT's calculate_depth method,
        # as it was not used for recursion anyway.
        # but we can keep it for the AoT baseline.
        if len(sub_qa_pairs) > 0:
            break

        # try:
        #     calculate_depth(sub_qa_pairs)
        #     result = {
        #         "answer": multistep_result["answer"],
        #         "response": multistep_result["response"],
        #         "sub-questions": sub_qa_pairs
        #     }
        #     break
        # except:
        #     retries -= 1
        #     continue
    return result


async def merging(model: str,
                  question: str,
                  decompose_result: dict,
                  independent: list[dict] = None,
                  dependent: list[dict] = None,
                  **kwargs):

    contractd_response = await contract(model=model,
                                        question=question,
                                        decompose_result=decompose_result,
                                        independent=independent,
                                        dependent=dependent,
                                        **kwargs)

    return contractd_response, parse_answer(contractd_response, key="question")


async def atom(model: str,
               question: str,
               cot_response: str = None,
               max_iters=1,
               log=None,
               **kwargs):
    # Initialize logging
    log = log if log else {}
    index = len(log)
    log[index] = {}

    refine_results = []
    for i in range(max_iters+1):

        # Get results from different approaches
        if i == 0 and cot_response is not None:
            cot_trajectory = cot_response
        else:
            cot_trajectory = (await direct(model=model, question=question, **kwargs))[0]

        # cot_trajectory = cot_trajectory[0]
        iteration_result = {
            "cot_response": cot_trajectory,
            "cot_answer": parse_answer(cot_trajectory)
        }

        # decompose with dependencies
        decompose_result = await decompose(model=model, question=question, trajectory=cot_trajectory, **kwargs)

        # The initial decompose result should be correctly logged for analysis.
        backup_decompose_result = copy.deepcopy(decompose_result)
        iteration_result["decompose_result"] = backup_decompose_result

        if i < max_iters:
            # Separate independent and dependent sub-questions
            independent_subqs = [
                sub_q for sub_q in decompose_result["sub-questions"]
                if "depend" not in sub_q or len(sub_q["depend"]) == 0
            ]
            dependent_subqs = [
                sub_q for sub_q in decompose_result["sub-questions"]
                if sub_q not in independent_subqs
            ]

            # Get contraction result
            contracted_thought, contracted_question = await merging(model=model,
                                                                    question=question,
                                                                    decompose_result=decompose_result,
                                                                    independent=independent_subqs,
                                                                    dependent=dependent_subqs,
                                                                    **kwargs)

            # Update contraction result with additional information
            iteration_result["contraction_thought"] = contracted_thought
            iteration_result["contraction_question"] = contracted_question

            question = contracted_question

        refine_results.append(iteration_result)

    candidate_responses = [result["cot_response"] for result in refine_results]

    # Get ensemble result
    ensemble_response = await ensemble(model=model,
                                       question=question,
                                       results=candidate_responses,
                                       **kwargs)
    ensemble_answer = parse_answer(ensemble_response)
    ensemble_result = {
        "response": ensemble_response,
        "answer": ensemble_answer
    }

    log[index].update({"refine_results": refine_results})

    return ensemble_result, log
