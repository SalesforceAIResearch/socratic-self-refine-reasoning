from experiment.utils import (
    parse_answer,
    min_confidence,
)

from experiment.module_atomic import (
    direct,
    decompose_and_confidence_estimate,
    socratic_refine,
    judge,
    refine,
    plan_summarize,
    refine_plan,
    direct_w_plan
)


async def socratic_self_refine(model: str,
                               question: str,
                               cot_response: str = None,
                               max_iter: int = 1,
                               log=None,
                               **kwargs):
    """
    Compared to the version v2, this version (v3) includes an additional round for the last-round refinement confidence estimation.
    """
    # Initialize logging
    log = log if log else {}
    index = len(log)
    log[index] = {}

    # Get the initial cot-trajectory
    from experiment.module_atomic import prompter
    direct_instruction = getattr(prompter, "direct")(
        model=model, question=question)

    iteration_results = []
    for i in range(max_iter+2):
        if i == 0 and cot_response is not None:
            cot_trajectory = cot_response
        elif i == 0 and cot_response is None:
            cot_trajectory = (await direct(model=model, question=question, **kwargs))[0]
        else:
            assert cot_trajectory is not None

        iteration_result = {
            "cot_response": cot_trajectory,
            "cot_answer": parse_answer(cot_trajectory)
        }
        # confidence estimation with socratic judge
        sub_qa_pairs, socratic_judge_results = await decompose_and_confidence_estimate(question=question,
                                                                                       cot_response=cot_trajectory,
                                                                                       model=model,
                                                                                       **kwargs)
        socratic_questions = [
            pair["description"] for pair in sub_qa_pairs]
        socratic_answers = [pair["answer"] for pair in sub_qa_pairs]
        socratic_answer_supports = [
            result for result in socratic_judge_results[0]]
        socratic_confidence_scores = [
            result for result in socratic_judge_results[1]]

        # If all confidence scores are 5 or no socratic questions, stop refinement
        if len(socratic_questions) == 0 or min_confidence(socratic_confidence_scores) == 5:
            # reduce to the case of normal self-refine,
            # even if the confidence scores are all 5.
            reflection = await judge(model=model,
                                     question=question,
                                     trajectory=cot_trajectory,
                                     **kwargs)
            # parse the confidence score.
            confidence_score = parse_answer(reflection, key="answer")
            if i == max_iter+1:
                refined_trajectory = cot_trajectory
            else:
                refined_trajectory = await refine(model=model,
                                                  question=question,
                                                  original_cot_response=cot_trajectory,
                                                  judge_response=reflection,
                                                  **kwargs)

            if min_confidence(socratic_confidence_scores) == 5:
                socratic_result = {
                    "socratic_questions": socratic_questions,
                    "socratic_answers": socratic_answers,
                    "socratic_answer_supports": socratic_answer_supports,
                    "socratic_confidence_scores": socratic_confidence_scores,
                }
            # This is a pseudo socratic result.
            else:
                socratic_result = {
                    "socratic_questions": [question],
                    "socratic_answers": [parse_answer(refined_trajectory)],
                    "socratic_answer_supports": [parse_answer(refined_trajectory)],
                    "socratic_confidence_scores": [confidence_score],
                }
        else:
            if i == max_iter+1:
                refined_trajectory = cot_trajectory
            else:
                refined_trajectory = await socratic_refine(direct_instruction=direct_instruction,
                                                           cot_trajectory=cot_trajectory,
                                                           socratic_questions=socratic_questions,
                                                           socratic_answers=socratic_answers,
                                                           socratic_answer_supports=socratic_answer_supports,
                                                           socratic_confidence_scores=socratic_confidence_scores,
                                                           model=model,
                                                           **kwargs)

            socratic_result = {
                "socratic_questions": socratic_questions,
                "socratic_answers": socratic_answers,
                "socratic_answer_supports": socratic_answer_supports,
                "socratic_confidence_scores": socratic_confidence_scores,
            }

        # parse the evaluation
        if "<evaluation>" in refined_trajectory and "</evaluation>" in refined_trajectory:
            cot_trajectory = refined_trajectory.split("</evaluation>")[1]
        else:
            cot_trajectory = refined_trajectory

        iteration_result["socratic_result"] = socratic_result
        iteration_result["refined_response"] = refined_trajectory
        iteration_result["refined_answer"] = parse_answer(refined_trajectory)
        iteration_results.append(iteration_result)

    log[index].update({"refine_results": iteration_results})

    return {"response": cot_trajectory, "answer": parse_answer(cot_trajectory)}, log


async def socratic_self_refine_adaptive(model: str,
                                        question: str,
                                        cot_response: str = None,
                                        max_iter: int = 1,
                                        log=None,
                                        **kwargs):
    """
    Compared to the version v2, this version (v3) includes an additional round for the last-round refinement confidence estimation.
    """
    # Initialize logging
    log = log if log else {}
    index = len(log)
    log[index] = {}

    # Get the initial cot-trajectory
    from experiment.module_atomic import prompter
    direct_instruction = getattr(prompter, "direct")(
        model=model, question=question)

    iteration_results = []
    for i in range(max_iter+2):
        if i == 0 and cot_response is not None:
            cot_trajectory = cot_response
        elif i == 0 and cot_response is None:
            cot_trajectory = (await direct(model=model, question=question, **kwargs))[0]
        else:
            assert cot_trajectory is not None

        iteration_result = {
            "cot_response": cot_trajectory,
            "cot_answer": parse_answer(cot_trajectory)
        }

        # start with the cheapest judge: normal self-refine.
        normal_reflection = await judge(model=model,
                                        question=question,
                                        trajectory=cot_trajectory,
                                        **kwargs)
        # parse the confidence score.
        normal_confidence_score = parse_answer(normal_reflection, key="answer")
        try:
            normal_confidence_score = int(normal_confidence_score)
        except:
            normal_confidence_score = -1

        iteration_result["normal_reflection"] = normal_reflection
        iteration_result["normal_confidence_score"] = normal_confidence_score

        # if normal judge is able to find the mistake,
        # then we directly use the normal refine
        if normal_confidence_score != 5:
            iteration_result["use_ssr"] = False
            socratic_result = None
            if i == max_iter+1:
                refined_trajectory = cot_trajectory
            else:
                refined_trajectory = await refine(model=model,
                                                  question=question,
                                                  original_cot_response=cot_trajectory,
                                                  judge_response=normal_reflection,
                                                  **kwargs)
        else:
            # if the confidence scores are all 5, or no socratic questions, stop refinement
            # confidence estimation with socratic judge
            sub_qa_pairs, socratic_judge_results = await decompose_and_confidence_estimate(question=question,
                                                                                           cot_response=cot_trajectory,
                                                                                           model=model,
                                                                                           **kwargs)
            socratic_questions = [
                pair["description"] for pair in sub_qa_pairs]
            socratic_answers = [pair["answer"] for pair in sub_qa_pairs]
            socratic_answer_supports = [
                result for result in socratic_judge_results[0]]
            socratic_confidence_scores = [
                result for result in socratic_judge_results[1]]

            # If all confidence scores are 5 or no socratic questions, stop refinement
            if len(socratic_questions) == 0 or min_confidence(socratic_confidence_scores) == 5:
                # reduce to the case of normal self-refine,
                # even if the confidence scores are all 5.
                iteration_result["use_ssr"] = False
                if i == max_iter+1:
                    refined_trajectory = cot_trajectory
                else:
                    refined_trajectory = await refine(model=model,
                                                      question=question,
                                                      original_cot_response=cot_trajectory,
                                                      judge_response=normal_reflection,
                                                      **kwargs)
                socratic_result = None
                if min_confidence(socratic_confidence_scores) == 5:
                    socratic_result = {
                        "socratic_questions": socratic_questions,
                        "socratic_answers": socratic_answers,
                        "socratic_answer_supports": socratic_answer_supports,
                        "socratic_confidence_scores": socratic_confidence_scores,
                    }

            else:
                iteration_result["use_ssr"] = True
                if i == max_iter+1:
                    refined_trajectory = cot_trajectory
                else:
                    refined_trajectory = await socratic_refine(direct_instruction=direct_instruction,
                                                               cot_trajectory=cot_trajectory,
                                                               socratic_questions=socratic_questions,
                                                               socratic_answers=socratic_answers,
                                                               socratic_answer_supports=socratic_answer_supports,
                                                               socratic_confidence_scores=socratic_confidence_scores,
                                                               model=model,
                                                               **kwargs)

                socratic_result = {
                    "socratic_questions": socratic_questions,
                    "socratic_answers": socratic_answers,
                    "socratic_answer_supports": socratic_answer_supports,
                    "socratic_confidence_scores": socratic_confidence_scores,
                }

        # parse the evaluation
        if "<evaluation>" in refined_trajectory and "</evaluation>" in refined_trajectory:
            cot_trajectory = refined_trajectory.split("</evaluation>")[1]
        else:
            cot_trajectory = refined_trajectory

        if model == "Qwen/Qwen3-Next-80B-A3B-Thinking":
            if "</think>" in cot_trajectory:
                cot_trajectory = cot_trajectory.split("</think>")[1]

        iteration_result["socratic_result"] = socratic_result
        iteration_result["refined_response"] = refined_trajectory
        iteration_result["refined_answer"] = parse_answer(refined_trajectory)
        iteration_results.append(iteration_result)

    log[index].update({"refine_results": iteration_results})

    return {"response": cot_trajectory, "answer": parse_answer(cot_trajectory)}, log


# new versions of the ssr, specifically for GPT-5 model.
async def socratic_self_refine_planning(model: str,
                                        question: str,
                                        cot_response: str = None,
                                        max_iter: int = 1,
                                        log=None,
                                        **kwargs):
    """
    Compared to the previous versions of SSR, this version (planning) will first judge whether the planning represented by the current CoT is correct. 
    CoT -> planning judge -> (Y) -> 3 x (Adaptive SSR).
    CoT -> planning judge -> (N) -> Refine the planning -> CoT -> 3 x (Adaptive SSR).
    """
    # Initialize logging
    log = log if log else {}
    index = len(log)
    log[index] = {}

    # dealing with the missing seed cot_response
    if cot_response is not None:
        seed_cot_response = cot_response
    else:
        seed_cot_response = (await direct(model=model, question=question, **kwargs))[0]

    if model == "Qwen/Qwen3-Next-80B-A3B-Thinking":
        # to address the issue of too-long cot context.
        if "</think>" in seed_cot_response:
            seed_cot_response = seed_cot_response.split("</think>")[1]

    plan = await plan_summarize(model=model,
                                question=question,
                                cot_response=seed_cot_response,
                                **kwargs)

    if model == "Qwen/Qwen3-Next-80B-A3B-Thinking":
        # to address the issue of too-long cot context.
        if "</think>" in plan:
            plan = plan.split("</think>")[1]

    plan_summary = parse_answer(plan, key="summary")
    plan_evaluation = parse_answer(plan, key="evaluation")
    try:
        plan_score = int(parse_answer(plan, key="score"))
    except:
        plan_score = -1

    if plan_score == 5:
        # this log will be automatically updated.
        refined_plan_response = None
        refined_plan = None
        refined_seed_cot_response = seed_cot_response
    else:
        refined_plan_response = await refine_plan(model=model,
                                                  question=question,
                                                  plan=plan_summary,
                                                  evaluation=plan_evaluation,
                                                  **kwargs)
        refined_plan = parse_answer(refined_plan_response, key="answer")
        refined_seed_cot_response = await direct_w_plan(model=model,
                                                        question=question,
                                                        plan=refined_plan,
                                                        **kwargs)
        if model == "Qwen/Qwen3-Next-80B-A3B-Thinking":
            # to address the issue of too-long cot context.
            if "</think>" in refined_seed_cot_response:
                refined_seed_cot_response = refined_seed_cot_response.split(
                    "</think>")[1]
    ssr_response, ssr_log = await socratic_self_refine_adaptive(model=model,
                                                                question=question,
                                                                cot_response=refined_seed_cot_response,
                                                                max_iter=max_iter,
                                                                log={},
                                                                **kwargs)

    log[index].update({
        "plan_result": {
            "seed_cot_response": seed_cot_response,
            "seed_cot_plan_summary": plan_summary,
            "seed_cot_plan_evaluation": plan_evaluation,
            "seed_cot_plan_score": plan_score,
            "refined_plan": refined_plan,
            "refined_seed_cot_response": refined_seed_cot_response,
        },
        "refine_results": ssr_log[0]["refine_results"]
    })

    return ssr_response, log
