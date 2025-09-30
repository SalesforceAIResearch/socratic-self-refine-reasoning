import copy
import math
import re
import random
from functools import lru_cache
import numpy as np

from experiment.module_atomic import call_llm_with_history as generate, call_llm
from experiment.utils import parse_answer
from .utils.utils import extract_boxed_answer, is_equiv


async def cal_reward(question,
                     ans,
                     model="gpt-4o-mini",
                     verbose=False,
                     **kwargs):
    from experiment.module_atomic import prompter
    messages = getattr(prompter, "mctsr_judge")(model=model,
                                                question=question,
                                                trajectory=ans)

    response = await call_llm(model=model, messages=messages, **kwargs)
    ret = response.choices[0].message.content
    if verbose:
        print("-" * 100)
        print(f"MCTSR Judge response for {question}:")
        print(ret)
        print("-" * 100)
    try:
        score = float(parse_answer(ret, key='answer'))
    except:
        score = 0
    return score


async def get_weak_answer(question,
                          new_len=0,
                          ans_format='',
                          model="gpt-4o-mini",
                          verbose=False,
                          **kwargs):
    from experiment.module_atomic import prompter
    instruction = getattr(prompter, "direct")(model=model,
                                              question=question)
    ret = await generate(instruction, model=model, **kwargs)
    if verbose:
        print("-" * 100)
        print(f"MCTSR Direct response for {question}:")
        print(ret[0])
        print("-" * 100)
    return ret[:2]


async def get_weak_hints(question,
                         weak_answer,
                         ground_truth_label=None,
                         new_len=0,
                         history=[],
                         alreadygood=False,
                         ans_format='',
                         model="gpt-4o-mini",
                         verbose=False,
                         **kwargs):
    from experiment.module_atomic import prompter
    instruction = getattr(prompter, "mctsr_feedback")(
        model=model, question=question)

    ret = await generate(instruction, history, model=model, **kwargs)
    if verbose:
        print("-" * 100)
        print(f"MCTSR Feedback response for {question}:")
        print(ret[0])
        print("-" * 100)
    return ret[:2]


async def get_better_answer(question,
                            weak_answer,
                            hint,
                            new_len=0,
                            history=[],
                            ans_format='',
                            model="gpt-4o-mini",
                            verbose=False,
                            **kwargs):
    from experiment.module_atomic import prompter
    query = getattr(prompter, "mctsr_refine")(
        model=model, question=question)
    ret = await generate(query, history, model=model, **kwargs)
    # TODO: delete the <evaluation> tags
    return ret[:2]


async def get_best_answer(question,
                          all_answer_str,
                          model="gpt-4o-mini",
                          verbose=False,
                          **kwargs):
    from experiment.module_atomic import prompter
    instruction = getattr(prompter, "ensemble")(model=model,
                                                question=question,
                                                solutions=all_answer_str)
    ret = await generate(instruction, model=model, **kwargs)
    if verbose:
        print("-" * 100)
        print(f"MCTSR Ensemble response for {question}:")
        print(ret[0])
        print("-" * 100)
    return ret


async def get_cot_answer(question, ans_format='', model="gpt-4o-mini", **kwargs):
    from experiment.module_atomic import prompter
    instruction = getattr(prompter, "direct")(model=model,
                                              question=question)
    ret = await generate(instruction, model=model, **kwargs)
    return ret[:1]


async def get_final_answer(query, model="gpt-4o-mini", **kwargs):
    ret = await generate(query, model=model, **kwargs)
    return ret[:2]


datas = []
pattern = re.compile(r'\-?\d+\.\d+|\-?\d+')


@lru_cache(1024)
def extract_label(DATA_NAME, text: str, type='') -> str:
    if 'gsm' not in DATA_NAME and type != 'digit':
        if '####' in text:
            text = text.split('####')[-1]
        elif 'The answer is' in text:
            text = text.split('The answer is')[-1]
            if '####' in text:
                text = text.split('####')[-1]
        if 'box' in text:
            return extract_boxed_answer(text)
        else:
            # remove the \n following the answer.
            if '\n' in text:
                text = text.split('\n')[0]
            return text
    if '\n####' in text:
        text = text.split('\n####')[-1].replace(',', '')
    elif 'The answer is' in text:
        text = text.split('The answer is')[-1].replace(',', '')
    if 'box' in text:
        return extract_boxed_answer(text)
    numbers = pattern.findall(text)
    if not numbers:
        return None
    if '\n####' in text or 'The answer is' in text:
        return numbers[0]
    else:
        return numbers[-1]


@lru_cache(1024)
def check(gt, ans, DATA_NAME):
    gt = str(gt)
    gt_label = extract_label(DATA_NAME, gt)
    if gt_label.isdigit():
        type = 'digit'
    elif gt_label.isupper() and gt_label.isalpha():
        type = 'option'
    elif gt_label.lower() in ['yes', 'no']:
        gt_label = gt_label.lower()
        type = 'yesorno'
    else:
        type = 'formula'
    ans_label = extract_label(DATA_NAME, ans)
    if ans_label:
        if type == 'option':
            ans_label = ans_label.strip()[0]
        elif type == 'yesorno':
            ans_label = ans_label.lower()
        elif type == 'formula':
            ans_label = ans_label.replace('$', '')
    print(gt_label, ans_label)
    if 'gsm' not in DATA_NAME and type != 'digit':
        return is_equiv(gt_label, ans_label)
    print(gt_label, ans_label)
    if gt_label is None or ans_label is None:
        return False
    try:
        if ans_label == gt_label or abs(float(ans_label) - float(gt_label)) < 1e-5:
            return True
        else:
            return False
    except:
        return False


def filter_mature_node(childs, to_explore, to_explore_reward, max_expand=3):
    filterd_to_explore = []
    avg_reward = {node: (min(
        to_explore_reward[node]) + np.mean(to_explore_reward[node])) / 2 for node in to_explore}

    for node in to_explore:
        if len(childs.get(node, [])) < max_expand or max([avg_reward.get(child, -999) for child in childs.get(node, [])]) < avg_reward.get(node, -999):
            filterd_to_explore.append(node)

    return filterd_to_explore


def get_best_explore_from_ucb(to_explore, ucb_bank):
    # 初始化最佳节点和最高UCB值
    best_node = None
    highest_ucb = float('-inf')

    # 遍历所有待探索的节点
    for node in to_explore:
        ucb_value = ucb_bank.get(node, float('-inf'))
        if ucb_value > highest_ucb:
            highest_ucb = ucb_value
            best_node = node

    return best_node


def compute_ucb(r_c, N_n, N_c, C):
    return r_c + C * math.sqrt(math.log(N_n + 1) / (N_c + 1e-5))


def update_ucb(fathers, childs, to_explore, to_explore_reward, ucb_bank, C=1.4, gamma=0.85):
    # 计算所有节点的访问次数
    visit_count = {node: len(to_explore_reward[node]) for node in to_explore}

    # 计算所有节点的平均奖励
    # avg_reward = {node: sum(to_explore_reward[node]) / len(to_explore_reward[node]) for node in to_explore}
    avg_reward = {node: (min(
        to_explore_reward[node]) + np.mean(to_explore_reward[node])) / 2 for node in to_explore}

    # 获取所有叶子节点
    leaves = set(to_explore) - set(fathers.values())

    # 更新所有叶子节点的UCB值
    for leaf in leaves:
        # ucb_bank[leaf] = avg_reward[leaf]
        ucb_bank[leaf] = compute_ucb(avg_reward[leaf], len(to_explore_reward.get(
            fathers.get(leaf, None), [])), len(to_explore_reward.get(leaf, [])), C)

    # 从叶子节点向上更新父节点的UCB值
    nodes_to_update = list(leaves)
    while nodes_to_update:
        new_nodes_to_update = set()
        for node in nodes_to_update:
            father = fathers.get(node)
            if father is not None:
                if father not in ucb_bank:
                    new_nodes_to_update.add(father)
                if father in ucb_bank:
                    # 计算父节点的UCB值
                    ucb_values = []
                    child_reward = []
                    for child in childs[father]:
                        ucb_values.append(ucb_bank[child])
                        child_reward.append(avg_reward[child])
                    father_reward = (avg_reward[father] + max(child_reward))/2
                    ucb_bank[father] = compute_ucb(father_reward, len(to_explore_reward.get(
                        fathers.get(father, None), [])), len(to_explore_reward.get(father, [])), C)
        nodes_to_update = list(new_nodes_to_update)


async def step(query,
               weak_answer,
               ground_truth_label=None,
               history=[],
               alreadygood=False,
               ans_format='',
               model="gpt-4o-mini",
               **kwargs):
    hints, history = await get_weak_hints(query,
                                          weak_answer,
                                          ground_truth_label=ground_truth_label,
                                          history=history,
                                          alreadygood=alreadygood,
                                          ans_format=ans_format,
                                          model=model,
                                          **kwargs)
    answer, history = await get_better_answer(query,
                                              weak_answer,
                                              hints,
                                              history=history,
                                              ans_format=ans_format,
                                              model=model,
                                              **kwargs)
    return hints, answer, history


def get_tree_ans(to_explore_reward, ucb_bank, answers_list):
    def weighted_score(answer):
        # 假设你根据奖励、访问次数、UCB等计算综合得分
        reward_score = min(to_explore_reward[answer]) * 0.5  # 奖励占50%
        visit_count = len(to_explore_reward[answer]) * 0.3  # 访问次数占30%
        ucb_score = ucb_bank.get(answer, 0) * 0.2  # UCB值占20%
        return reward_score + visit_count + ucb_score

    best_answer = max(answers_list, key=weighted_score)
    return best_answer


async def monte_carlo_tree(DATA_NAME, query, max_iter=16, ans_format='', model="gpt-4o-mini", **kwargs):
    is_activate = True

    to_explore = []
    to_explore_reward = {}
    history_bank = {}
    hints_bank = {}
    ucb_bank = {}
    fathers = {}
    childs = {}

    async def sampling_reward(answer, **kwargs):
        if answer not in to_explore_reward:
            to_explore_reward[answer] = []
        reward = await cal_reward(query, answer, model=model, **kwargs)
        to_explore_reward[answer].append(reward)

    def add_to_hints_bank(hints, weak_answer):
        if weak_answer not in hints_bank:
            hints_bank[weak_answer] = []
        hints_bank[weak_answer].append(hints)

    def add_to_childs(father, child):
        if father not in childs:
            childs[father] = []
        childs[father].append(child)

    hints_reward_imp_bank = {}

    def add_to_hints_reward_imp_bank(hints, weak_answer, reward, answer):
        if weak_answer not in hints_reward_imp_bank:
            hints_reward_imp_bank[weak_answer] = []
        hints_reward_imp_bank[weak_answer].append((hints, reward, answer))

    ### get weak answer###
    weak_answer, history = await get_weak_answer(query, ans_format=ans_format, model=model, **kwargs)

    history_bank[weak_answer] = tuple(history)
    activated_answers_list = []
    activated_answer_scores = []

    answers_list = [weak_answer,]
    to_explore = [weak_answer,]
    childs[weak_answer] = []
    fathers[weak_answer] = None

    await sampling_reward(weak_answer, **kwargs)
    activated_answer_scores.append(to_explore_reward[weak_answer])
    # TODO: double check if this is correct
    if to_explore_reward[weak_answer][-1] > 95:
        return weak_answer, is_activate, activated_answers_list, activated_answer_scores, answers_list, to_explore, to_explore_reward, ucb_bank

    total_bad = random.choice(["I Don't Know", "I can't understand this question.", "I can't help with this question.",
                              "I don't know how to solve this question.", "I don't know the answer to this question.", "I don't know the answer to this question, sorry."])
    total_bad_history = copy.deepcopy(history)
    total_bad_history[-1] = total_bad
    history_bank[total_bad] = tuple(total_bad_history)

    answers_list += [total_bad,]
    to_explore += [total_bad,]
    childs[total_bad] = []
    fathers[total_bad] = None
    await sampling_reward(total_bad, **kwargs)
    update_ucb(fathers=fathers, childs=childs, to_explore=to_explore,
               to_explore_reward=to_explore_reward, ucb_bank=ucb_bank)

    for i in range(max_iter):
        # print('iteration:', i)
        filterd_to_explore = filter_mature_node(
            childs, to_explore, to_explore_reward)
        weak_answer = get_best_explore_from_ucb(
            filterd_to_explore, ucb_bank)  # selection
        await sampling_reward(weak_answer, **kwargs)  # similation
        # extend
        hints, answer, history = await step(
            query, weak_answer, history=history_bank[weak_answer], ans_format=ans_format, model=model, **kwargs)
        add_to_hints_bank(hints, weak_answer)
        history_bank[answer] = tuple(history)
        to_explore.append(answer)
        await sampling_reward(answer, **kwargs)
        fathers[answer] = weak_answer
        childs[answer] = []
        add_to_childs(weak_answer, answer)

        # extract_ans = extract_label(DATA_NAME, answer)
        extract_ans = parse_answer(answer, key='answer')
        if extract_ans:
            activated_answers_list.append(extract_ans)
            activated_answer_scores.append(to_explore_reward[answer])
        answers_list.append(answer)

        update_ucb(fathers=fathers, childs=childs, to_explore=to_explore,
                   to_explore_reward=to_explore_reward, ucb_bank=ucb_bank)
        add_to_hints_reward_imp_bank(hints, weak_answer, min(to_explore_reward.get(answer)) - min(
            to_explore_reward.get(weak_answer)), answer)  # ucb_bank[answer] - ucb_bank[weak_answer]

    tree_ans = get_tree_ans(to_explore_reward, ucb_bank, answers_list)
    return tree_ans, is_activate, activated_answers_list, activated_answer_scores, answers_list, to_explore, to_explore_reward, ucb_bank
