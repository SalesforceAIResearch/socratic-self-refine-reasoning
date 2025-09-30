################################################################################
# Chain-of-Thought (CoT)
################################################################################

def direct(model: str, question: str):
    if "gpt-5" in model.lower():
        instruction = """You are a precise math problem solver. Solve the given math problem step by step:

QUESTION: {question}

You can freely reason in your response, please: 
1. include the thinking process in your response, in a detailed and extended manner, and after that, 
2. enclose the final answer within <answer></answer> tags (pure number without units and explanations)
"""
    else:
        instruction = """You are a precise math problem solver. Solve the given math problem step by step:

QUESTION: {question}

Please extend your chain of thought as much as possible; the longer the chain of thought, the better.

You can freely reason in your response, but please enclose the final answer within <answer></answer> tags (pure number without units and explanations)
"""

    prompt = instruction.format(question=question)
    return prompt


def multistep(model: str, question: str):
    return direct(model, question)


################################################################################
# Atom of Thought (AOT)
################################################################################

def decompose_w_dependencies(model: str, question: str, trajectory: str, answer: str):
    """
    This function (previously called `label` in AoT paper) has several improvements over its old version:
        1. We change the intermediate answer to expression from a single numerical answer.
        2. the sub-questions are vague and hard to answer.
        3. after the final question leading to the final answer, there are some redundant sub-questions that makes the LLM hallucinate incorrect answers.
        4. the answers of the intermediate sub-question have approximation error (e.g., frac{1}{3} is converted to 0.33), and the accumulation of such approximation error leads to the incorrect final answer.
    """
    instruction = """You are tasked with breaking down a math problem's reasoning process into a series of sub-questions.

Original Question: {question}
Complete Reasoning Process: {trajectory}

Instructions:
1. Break down the reasoning process into a series of sub-questions.
2. Each sub-question should:
    - Be written in a clear, interrogative form.
    - Be precise, unambiguous, and directly answerable from the provided reasoning or prior sub-question answers.
    - Have a clear, **exact expression** as its answer (e.g., use fractions like `1/3`, symbolic representations like `pi`, or precise numerical values such as `1.0`). **Crucially, avoid approximations or rounding** unless the original question explicitly requires it.
    - List the 0-based indexes of other sub-questions it depends on. This list can be empty if no prior sub-question answers are needed.
3. Dependencies are defined as information necessary to answer the current sub-question that:
    - Does NOT come directly from the original question.
    - MUST come from the answers of previous sub-questions.
4. **Stop generating sub-questions once the final answer to the Original Question has been fully derived from the reasoning process.** Do not include any subsequent or irrelevant steps that do not directly contribute to reaching the final answer.
"""
    formatter = """Format your response as the following JSON object:
{{
    "sub-questions": [
        {{
            "description": "<clear, precise interrogative question>",
            "answer": <exact expression of the answer, avoiding approximations>,
            "depend": [<indices of prerequisite sub-questions>]
        }},
        ...
    ],
    "answer": {answer}
}}
"""
    return (instruction + formatter).format(question=question,
                                            trajectory=trajectory,
                                            answer=answer)


def contract(model: str,
             question: str,
             decompose_result: dict,
             independent=None,
             dependent=None) -> str:
    instruction = """You are a math problem solver specializing in optimizing step-by-step reasoning processes. Your task is to optimize the existing reasoning trajectory into a more efficient, single self-contained question.
        
For the original question: {question}

Here are step-by-step reasoning process:
{response}

{sub_questions}

Here are explanations of key concepts:
1. self-contained: The optimized question must be solvable independently, without relying on any external information
2. efficient: The optimized question must be simpler than the original, requiring fewer reasoning steps (these steps are reduced because some solved independent sub-problems become known conditions in the optimized question or are excluded as incorrect explorations)

You can freely reason in your response, but please enclose the your optimized question within <question></question> tags
"""

    independent_sub_questions = """
The following sub-questions and their answers can serve as known conditions:
{independent}
"""

    dependent_sub_questions = """
The descriptions of the following questions can be used to form the description of the optimized problem:
{dependent}    
"""
    answer = decompose_result["answer"]

    if independent not in [None, []]:
        for sub_q in independent:
            sub_q.pop("depend", None)
    if dependent is not None:
        for sub_q in dependent:
            sub_q.pop("depend", None)

    if independent not in [None, []]:
        sub_questions = independent_sub_questions.format(
            independent=independent) + dependent_sub_questions.format(
                dependent=dependent)
    elif independent is not None:
        sub_questions = independent_sub_questions.format(
            independent=independent)
    else:
        sub_questions = ""
    return instruction.format(question=question,
                              answer=answer,
                              response=decompose_result["response"],
                              sub_questions=sub_questions)


def ensemble(model: str, question: str, solutions: list):
    instruction = """You are a precise math problem solver. Compare then synthesize the best answer from multiple solutions to solve the following question.

QUESTION: {question}

SOLUTIONS:
{solutions}

Please extend your chain of thought as much as possible; the longer the chain of thought, the better.

You can freely reason in your response, but please enclose the final answer within <answer></answer> tags (pure number without units and explanations)
"""

    solutions_str = ""
    for i, solution in enumerate(solutions):
        solutions_str += f"solution {i}: {solution}\n"
    prompt = instruction.format(question=question, solutions=solutions_str)
    return prompt


################################################################################
# Self-Refine
################################################################################

def judge(model: str, question: str, trajectory: str):
    PROMPT_VERIFY_SYSTEM = """
Please act as an impartial judge and evaluate the correctness of the response provided by an AI assistant to the user prompt displayed below. You will be given the assistant's response.

When evaluating the assistant's response, identify any mistakes or inaccurate information. Be as objective as possible. Avoid any biases, such as order of responses, length, or stylistic elements like formatting.

Before providing an your final verdict, think through the judging process and output your thoughts as an explanation

After providing your explanation, you must output a score of scale 0 to 5, where 0 represents you are completely certain that the response is incorrect and 5 represents you are completelycertain that the response is correct. Please enclose your score in <answer> and </answer> tags.
""".strip()

    PROMPT_VERIFY = """
<|User Prompt|>
{question}
<|The Start of Assistant's Answer|>
{response}
<|The End of Assistant's Answer|>
""".strip()

    def render_verify_prompt(response, question, prompt_strategy='vanilla'):
        if prompt_strategy == 'vanilla':
            sys_prompt = PROMPT_VERIFY_SYSTEM
        prompt_template = PROMPT_VERIFY
        prompt_formatted = prompt_template.format(question=question,
                                                  response=response)
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt_formatted}]
        return messages

    return render_verify_prompt(trajectory, question, prompt_strategy='vanilla')


def refine(model: str, question: str, original_cot_response: str, judge_response: str):
    instruction = """You are a precise math problem solver. Refine the provided solution to the given math problem, step-by-step, by meticulously addressing the judge's feedback.

**QUESTION:** {question}
**ORIGINAL SOLUTION:** {original_cot_response}
**JUDGE RESPONSE:** {judge_response}

Your task is to re-evaluate the original reasoning, identify where it went wrong based on the judge's comments, which should be enclosed within <evaluation></evaluation> tags; after that, construct a new, corrected chain of thought. Explain each step thoroughly. The more detailed and explicit your reasoning, the better.

You can freely reason in your response, but please enclose the final, numerical answer within <answer></answer> tags (pure number only, without units or explanations).
"""

    prompt = instruction.format(question=question,
                                original_cot_response=original_cot_response,
                                judge_response=judge_response)
    return prompt


################################################################################
# LLM-Debate (Debate)
################################################################################

def debate_refine(model: str, question: str, original_cot_response: str, other_agents_responses: list):
    instruction = """You are a precise math problem solver. Using the solutions from other agents as additional information, refine your original solution to the given math problem, step-by-step.

**QUESTION:** {question}
**ORIGINAL SOLUTION:** {original_cot_response}
**OTHER AGENTS' SOLUTIONS:** {other_agents_responses}

You can freely reason in your response, but please enclose the final, numerical answer within <answer></answer> tags (pure number only, without units or explanations).
"""
    prompt = instruction.format(question=question,
                                original_cot_response=original_cot_response,
                                other_agents_responses="\n".join([f"Agent {i}: {response}" for i, response in enumerate(other_agents_responses)]))
    return prompt


################################################################################
# Monte Carlo Tree Self-refine (MCTSr)
################################################################################

def mctsr_judge(model: str, question: str, trajectory: str):
    PROMPT_VERIFY_SYSTEM = """
Please act as an impartial judge and evaluate the correctness of the response provided by an AI assistant to the user prompt displayed below. You will be given the assistant's response.

When evaluating the assistant's response, identify any mistakes or inaccurate information. Be as objective as possible. Avoid any biases, such as order of responses, length, or stylistic elements like formatting.

Before providing an your final verdict, think through the judging process and output your thoughts as an explanation

After providing your explanation, you must output a score of scale 0 to 100, where 0 represents you are completely certain that the response is incorrect and 100 represents you are completely certain that the response is correct. Please enclose your score in <answer> and </answer> tags.
""".strip()

    PROMPT_VERIFY = """
<|User Prompt|>
{question}
<|The Start of Assistant's Answer|>
{response}
<|The End of Assistant's Answer|>
""".strip()

    def render_verify_prompt(response, question, prompt_strategy='vanilla'):
        if prompt_strategy == 'vanilla':
            sys_prompt = PROMPT_VERIFY_SYSTEM
        prompt_template = PROMPT_VERIFY
        prompt_formatted = prompt_template.format(question=question,
                                                  response=response)
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt_formatted}]
        return messages

    return render_verify_prompt(trajectory, question, prompt_strategy='vanilla')


def mctsr_feedback(model: str, question: str):
    instruction = """
Since we have a weak answer, could you provide me with a relection or feedback to correct this answer better? Please act as an impartial judge and evaluate the correctness of the response provided by an AI assistant to the user prompt displayed below. 

**QUESTION:** {question}

When evaluating the assistant's response, identify any mistakes or inaccurate information. Be as objective as possible. Avoid any biases, such as order of responses, length, or stylistic elements like formatting.Before providing an your final verdict, think through the judging process and output your thoughts as an explanation.
""".strip().format(question=question)

    return instruction


def mctsr_refine(model: str, question: str):
    instruction = """Refine the provided solution to the given math problem, step-by-step, by meticulously addressing the judge's feedback as above.

**QUESTION:** {question}

Your task is to re-evaluate the original reasoning, identify where it went wrong based on the judge's comments, which should be enclosed within <evaluation></evaluation> tags; after that, construct a new, corrected chain of thought. Explain each step thoroughly. The more detailed and explicit your reasoning, the better.

You can freely reason in your response, but please enclose the final, numerical answer within <answer></answer> tags.""".format(question=question)

    return instruction


################################################################################
# Linear Socratic Self-Refine (SSR-Lin)
################################################################################

def decompose(model: str, question: str, cot_response: str):

    instruction = """You are tasked with breaking down a math problem's reasoning process into a series of **atomic** sub-questions.

Original Question: {question}
Complete Reasoning Process: {trajectory}

Instructions:
1. Break down the reasoning process into a series of sub-questions.
2. Each sub-question should:
    - Be written in a clear, interrogative form.
    - Be precise, unambiguous, and directly answerable from the provided reasoning or prior sub-question answers.
    - Have a clear, **exact expression** as its answer (e.g., use fractions like `1/3`, symbolic representations like `pi`, or precise numerical values such as `1.0`). **Crucially, avoid approximations or rounding** unless the original question explicitly requires it.
3. **Stop generating sub-questions once the final answer to the Original Question has been fully derived from the reasoning process.** Do not include any subsequent or irrelevant steps that do not directly contribute to reaching the final answer.
4. The sub-question, sub-answer pairs should perfectly represent the reasoning process of the solution.
"""
    if "llama" in model.lower():
        formatter = """Format your response as the following JSON object:
{{
    "sub-questions": [
        "<clear, precise interrogative question 1>",
        ...
    ],
    "sub-answers": [
        "<exact expression of the answer 1, avoiding approximations>",
        ...
    ]
}}
"""
    else:
        formatter = """Format your response as the following JSON object:
{{
    "sub-questions": [
        {{
            "description": "<clear, precise interrogative question>",
            "answer": <exact expression of the answer, avoiding approximations>,
        }},
        ...
    ]
}}
"""
    prompt = (instruction + formatter).format(question=question,
                                              trajectory=cot_response)
    return prompt


def socratic_answer_next_sub_question(model: str,
                                      question: str,
                                      sub_question_answer_pairs: list,
                                      next_sub_question: str):
    socratic_reasoning_trajectory = "\n".join(
        [f"Q{i+1}. {sub_question['description']}\nA{i+1}. {sub_question['answer']}" for i, sub_question in enumerate(sub_question_answer_pairs)])

    system_prompt = """You are a precise math problem solver. Given the original question and the series of sub-questions and their answers which perfectly represent the reasoning process of the solution, think step by step and answer the next sub-question. Do not extend the reasoning process beyond this sub-question and enclose the answer within <answer></answer> tags."""
    instruction = """Original question:
{question}

The series of sub-questions and their answers:
{socratic_reasoning_trajectory}

The next sub-question to be answered:
{next_sub_question}
"""
    user_prompt = instruction.format(
        question=question, socratic_reasoning_trajectory=socratic_reasoning_trajectory, next_sub_question=next_sub_question)
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]


def check_equivalence(model: str, prediction: str, reference_answers: list):
    instruction = """You are a math expert. Given the a math expression as the prediction and a list of reference answers, determine the confidence of the prediction.
The prediction is:
{prediction}

The reference answers are:
{answers}

Please answer with a number of scale 0 to 5 that represents the confidence of the prediction. 0 means the prediction does not match any of the reference answers. 5 means the prediction matches the reference answers perfectly. If you cannot determine the confidence, please answer with -1. Enclose the answer within <answer></answer> tags."""

    return instruction.format(prediction=prediction,
                              answers="\n".join([f"{chr(ord('A')+i)}. {answer}" for i, answer in enumerate(reference_answers)]))


def socratic_reflection(model: str,
                        wrong_question: str,
                        wrong_answer: str,
                        revised_answer: str):
    return """Wait, in the sub-step of "{wrong_question}", the answer is "{wrong_answer}", but after careful re-evaluating the process, I think that the actual answer to this sub-question should be "{revised_answer}".""".format(
        wrong_question=wrong_question, wrong_answer=wrong_answer, revised_answer=revised_answer)


def socratic_refine(model: str,
                    direct_instruction: str,
                    cot_trajectory: str,
                    reflection: str):
    if "llama" in model.lower():
        re_eval_content = "Let's re-evaluate the reasoning process, which should be enclosed within <evaluation></evaluation> tags; after that, address the specific issue raised in the re-evaluation and improve the reasoning process, enclose the final answer within <answer></answer> tags."
    else:
        re_eval_content = "Let's re-evaluate the reasoning process based on your reflection. Enclose it within <evaluation></evaluation> tags. After that, let's reasoning step by step again to solve the original question. This time, you should address the specific issue identified in your own re-evaluation. Finally,enclose the final answer within <answer></answer> tags."

    messages = [
        {"role": "user", "content": direct_instruction},
        {"role": "assistant", "content": cot_trajectory + "\n\n" + reflection},
        {"role": "user", "content": re_eval_content}
    ]
    return messages


################################################################################
# Socratic Self-Refine w/ Plan Refinement (SSR-Plan)
################################################################################

def plan_summarize(model: str, question: str, cot_response: str):
    return """You are a precise and knowledgeable math problem solver. Your task is to: 

1. Summarize high-level planning of the reasoning process of the following solution to the given math problem. NOTE: the summary of the planning should be concise, clear, and **not** including any specific calculations. 
2. Evaluate the quality of the planning (not the solution itself). 

Please enclose the summary of the planning within <summary></summary> tags, the evaluation of the planning within <evaluation></evaluation> tags, and provide the score (evaluation of the planning quality) within <score></score> tags (on scale 0 to 5, with 5 being the highest quality, 0 being the lowest quality).

**QUESTION:** {question}
**REASONING PROCESS:** {cot_response}
""".format(question=question, cot_response=cot_response)


def refine_plan(model: str, question: str, plan: str, evaluation: str):
    return """You are a precise and knowledgeable math problem solver. Refine the following high-level  plan to the given math problem based on the evaluation of the planning quality.

**QUESTION:** {question}
**CURRENT PLAN:** {plan}
**EVALUATION OF THE PLANNING QUALITY:** {evaluation}

You can freely reason in your response, but please enclose the final refined plan within <answer></answer> tags.
""".format(question=question, plan=plan, evaluation=evaluation)


def direct_w_plan(model: str, question: str, plan: str):
    if "gpt-5" in model.lower():
        instruction = """You are a precise math problem solver. Solve the given math problem step by step based on the following plan:

QUESTION: {question}
PLAN: {plan}

You can freely reason in your response, please: 
1. include the thinking process in your response, in a detailed and extended manner, and after that, 
2. enclose the final answer within <answer></answer> tags (pure number without units and explanations)
"""
    else:
        instruction = """You are a precise math problem solver. Solve the given math problem step by step based on the following plan:

QUESTION: {question}
PLAN: {plan}

Please extend your chain of thought as much as possible; the longer the chain of thought, the better.

You can freely reason in your response, but please enclose the final answer within <answer></answer> tags (pure number without units and explanations)
"""

    prompt = instruction.format(question=question, plan=plan)
    return prompt
