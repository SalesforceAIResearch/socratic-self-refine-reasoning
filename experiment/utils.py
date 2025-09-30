from textwrap import wrap
import matplotlib.pyplot as plt
import networkx as nx
import re
import os
import json
import re
import string
from collections import Counter
from typing import List, Union
import math
from math_verify import parse, verify
import warnings
import numpy as np
import re


def extract_json(string):
    try:
        string = string.strip()
        start, end = string.find("{"), string.rfind("}")
        if start != -1 and end != -1:
            string = string[start:end + 1]
        json_data = json.loads(string)
        return json_data
    except Exception as e:
        return str(e)


def extract_xml(string):
    try:
        # Remove any leading/trailing whitespace
        string = string.strip()

        # Use regex to find all tag-content pairs
        pattern = r"<([\w-]+)>(.*?)</\1>"
        matches = re.finditer(pattern, string)

        result = {}

        # Process each match, later matches will overwrite earlier ones
        for match in matches:
            tag = match.group(1)
            content = match.group(2).strip()

            # Try to convert content to number if possible
            try:
                if content.isdigit():
                    value = int(content)
                else:
                    value = float(content)
            except:
                value = content

            # Simply update the value, overwriting any previous value
            result[tag] = value

        return result
    except Exception as e:
        return {}


def check_json(json_obj, keys: list):
    if not isinstance(json_obj, dict):
        return False
    for key in keys:
        if key not in json_obj.keys():
            return False
    return True


def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}


def duration_formatter(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    if hours > 0:
        return f"{int(hours):02d}h:{int(minutes):02d}m:{int(seconds):02d}s"
    elif minutes > 0:
        return f"{int(minutes):02d}m:{int(seconds):02d}s"
    else:
        return f"{int(seconds):02d}s"


def calculate_depth(sub_questions: list):
    try:
        n = len(sub_questions)

        # Initialize distances matrix with infinity
        distances = [[float("inf")] * n for _ in range(n)]

        # Set direct dependencies
        for i, sub_q in enumerate(sub_questions):
            # Distance to self is 0
            distances[i][i] = 0
            # Set direct dependencies with distance 1
            for dep in sub_q.get("depend", []):
                distances[dep][i] = 1

        # Floyd-Warshall algorithm to find shortest paths
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if distances[i][k] != float(
                            "inf") and distances[k][j] != float("inf"):
                        distances[i][j] = min(
                            distances[i][j], distances[i][k] + distances[k][j])

        # Find maximum finite distance
        max_depth = 0
        for i in range(n):
            for j in range(n):
                if distances[i][j] != float("inf"):
                    max_depth = max(max_depth, distances[i][j])

        return int(max_depth)
    except:
        return 3


def get_next_log_file(log_dir, model, size, dataset):
    directory = log_dir.format(model=model, dataset=dataset, size=size)
    os.makedirs(directory, exist_ok=True)

    # 只计算数字命名的json文件，排除score
    files = [
        f for f in os.listdir(directory)
        if f.endswith('.json') and f != 'score.json'
    ]

    # 找出最大的数字编号
    max_num = 0
    for f in files:
        try:
            num = int(f.split('.')[0])
            max_num = max(max_num, num)
        except ValueError:
            continue

    return os.path.join(directory, f"{max_num + 1}.json")


def get_file_count(log_dir, model, interval, dataset, exclude_score=False):
    directory = log_dir.format(model=model, dataset=dataset, size=interval)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return 0

    files = os.listdir(directory)
    if exclude_score:
        # 排除score.json，只计算数字命名的json文件
        files = [f for f in files if f != "score.json"]

    return len(files)


# hotpotqa
def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def score_zebra(prediction, ground_truth):
    prediction = re.sub(r'[^a-zA-Z]', '', prediction)
    ground_truth = re.sub(r'[^a-zA-Z]', '', ground_truth)
    return exact_match_score(prediction, ground_truth)


def sudoku_match_score(prediction, ground_truth: dict):
    # this ground truth is actually the meta data, only for sudoku dataset.
    # solution:
    gt_string = "".join([str(x)
                        for row in ground_truth["solution"] for x in row])
    pred_string = re.sub(r'[^\d]', '', prediction)
    if gt_string == pred_string:
        return 1

    puzzle = ground_truth["puzzle"]
    try:
        prediction = [row.split() for row in prediction.strip().split("\n")]
        # check if the prediction obeys the puzzle constraints
        for i in range(4):
            for j in range(4):
                if int(prediction[i][j]) != int(puzzle[i][j]) and int(puzzle[i][j]) != 0:
                    return 0
        # check each column and each 2x2 subgrid
        # Check each row
        for i in range(4):
            row = prediction[i]
            if sorted(row) != ['1', '2', '3', '4']:
                return 0
        # Check each column
        for j in range(4):
            col = [prediction[i][j] for i in range(4)]
            if sorted(col) != ['1', '2', '3', '4']:
                return 0
        # Check each 2x2 subgrid
        for block_i in [0, 2]:
            for block_j in [0, 2]:
                subgrid = []
                for i in range(block_i, block_i + 2):
                    for j in range(block_j, block_j + 2):
                        subgrid.append(prediction[i][j])
                if sorted(subgrid) != ['1', '2', '3', '4']:
                    return 0
    except:
        return 0
    return 1


def cryptarithm_match_score(prediction, ground_truth):
    prediction = prediction.replace("\n", "").replace(" ", "")
    ground_truth = ground_truth.replace("\n", "").replace(" ", "")

    def extract_dict(s):
        return {k: v for k, v in [row.split("=") for row in s.split(",") if "=" in row]}

    def match_dict(pred_dict, gt_dict):
        for k, v in pred_dict.items():
            if k not in gt_dict or gt_dict[k] != v:
                return 0
        for k, v in gt_dict.items():
            if k not in pred_dict or pred_dict[k] != v:
                return 0
        return 1

    try:
        return match_dict(extract_dict(prediction), extract_dict(ground_truth))
    except:
        return 0


def cryptarithm_match_score_iou(prediction, ground_truth):
    prediction = prediction.replace("\n", "").replace(" ", "")
    ground_truth = ground_truth.replace("\n", "").replace(" ", "")

    def extract_dict(s):
        return {k: v for k, v in [row.split("=") for row in s.split(",") if "=" in row]}

    def IoU(pred_dict, gt_dict):
        correct, total = 0, 0
        for k, v in gt_dict.items():
            total += 1
            if k in pred_dict and pred_dict[k] == v:
                correct += 1
        return correct / (total + len(pred_dict) - correct)

    try:
        return IoU(extract_dict(prediction), extract_dict(ground_truth))
    except:
        return 0


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if (normalized_prediction in ["yes", "no", "noanswer"]
            and normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC
    if (normalized_ground_truth in ["yes", "no", "noanswer"]
            and normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def score_mh(prediction: str, groundtruth: Union[str, list]):
    try:
        if isinstance(groundtruth, list):
            f1 = max([f1_score(prediction, gt)[0] for gt in groundtruth])
        else:
            f1 = f1_score(prediction, groundtruth)[0]
        return f1
    except:
        return 0


# math
def extract_boxed(s):
    import re

    pattern = r"\\boxed{((?:[^{}]|{(?:[^{}]|{[^{}]*})*?)*)}"
    match = re.search(pattern, s)
    if match:
        return match.group(1)
    return ""


def eval_math(s):
    try:
        return eval(str(s).replace(",", ""))
    except:
        return 0


def score_math(prediction, groundtruth, dataset="aime", verbose=False):
    # in FoT and MCTSr, the answer sometimes contains noisy characters.
    if isinstance(prediction, str) and "\n" in prediction:
        if verbose:
            warnings.warn(
                f"Prediction contains noisy characters: {prediction}")
        prediction = prediction.split("\n")[0]
        if verbose:
            warnings.warn(f"Prediction after cleaning: {prediction}")

    # true verification.
    flag = 0
    try:
        if dataset.startswith("math"):
            gt, pd = eval_math(extract_boxed(groundtruth)
                               ), eval_math(prediction)
            flag = 1 if pd == gt else 0
        elif dataset == "gsm8k":
            gt, pd = eval_math(groundtruth.split("####")[
                               1]), eval_math(prediction)
            flag = 1 if pd == gt else 0
        elif dataset == "gsm_hard":
            gt, pd = eval_math(groundtruth), eval_math(prediction)
            flag = 1 if math.isclose(pd, gt, rel_tol=1e-6) else 0
        elif dataset.startswith("aime"):
            gt, pd = eval_math(groundtruth), eval_math(prediction)
            flag = 1 if pd == gt else 0

        # we have another layer of verification if the previous naive verification fails.
        if flag:
            return flag
        flag = 1 if verify(parse(pd), parse(gt)) else 0
    except:
        try:
            flag = 1 if verify(parse(pd), parse(gt)) else 0
        except:
            pass
    return flag


# logic
def score_mc(prediction, target):
    if not prediction or not target:
        return 0

    prediction = str(prediction).upper()
    target = str(target).upper()

    def normalize_answer(answer):
        # Remove any brackets and convert to uppercase
        return answer.replace("(", "").replace(")", "").upper()

    return 1 if normalize_answer(prediction) == normalize_answer(target) else 0


########################################################################################
# Socratic Reasoning Utils.
########################################################################################

def cost_estimate(model, num_input_tokens, num_output_tokens):
    if model == "gpt-4o-mini":
        price_input = 0.15
        price_output = 0.6
    elif model == "gpt-4o":
        price_input = 2.5
        price_output = 10
    elif model == "gpt-4.1":
        price_input = 2
        price_output = 8
    elif model == "gpt-4.1-mini":
        price_input = 0.4
        price_output = 1.6
    elif model == "gpt-4.1-nano":
        price_input = 0.1
        price_output = 0.4
    elif model == "gpt-5-nano":
        price_input = 0.05
        price_output = 0.4
    elif model == "gpt-5-mini":
        price_input = 0.25
        price_output = 2
    elif model == "google/gemini-2.5-flash-lite":
        price_input = 0.1
        price_output = 0.1
    elif model == "google/gemini-2.5-flash":
        price_input = 0.5
        price_output = 0.5
    else:
        print(f"Model {model} not supported")
        return -1

    return (price_input * num_input_tokens + price_output * num_output_tokens) / 1e6


def parse_answer(response, key: str = "answer"):
    """By default, we parse the last answer in the response."""
    if f"<{key}>" in response and f"</{key}>" in response:
        # Parse the last answer enclosed in <answer>...</answer>
        import re
        matches = re.findall(f"<{key}>(.*?)</{key}>", response, re.DOTALL)
        if matches:
            return matches[-1].strip()
    elif "\\boxed{" in response and "}" in response:
        import re
        matches = re.findall(r"\\boxed\{(.*?)\}", response, re.DOTALL)
        if matches:
            return matches[-1].strip()
    else:
        return "NA"


def confidence_scores_format_check(func):
    def wrapper(confidence_scores):
        if isinstance(confidence_scores, list):
            confidence_scores = np.array([float(x) for x in confidence_scores])
        if len(confidence_scores) == 0:
            confidence_scores = np.array([0])
        return func(confidence_scores)
    return wrapper


@confidence_scores_format_check
def logprob(confidence_scores):
    confidence_scores[confidence_scores < 0] = 0
    return np.log(confidence_scores/5 + 1e-8).sum()


@confidence_scores_format_check
def mean_logprob(confidence_scores):
    confidence_scores[confidence_scores < 0] = 0
    return np.log(confidence_scores/5 + 1e-8).mean()


@confidence_scores_format_check
def mean_confidence(confidence_scores):
    return confidence_scores.mean()


@confidence_scores_format_check
def min_confidence(confidence_scores):
    return confidence_scores.min()


@confidence_scores_format_check
def non_zero_confidence(confidence_scores):
    return 0 if any(confidence_scores == 0) else 1


AGGREGATION_METHODS_MAPPING = {
    "logprob": logprob,
    "mean_logprob": mean_logprob,
    "mean_confidence": mean_confidence,
    "min_confidence": min_confidence,
    "non_zero_confidence": non_zero_confidence
}


def majority_vote_answers(answers):
    return Counter(answers).most_common(1)[0][0]
