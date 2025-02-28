import json
import os
import random
import re
from typing import Dict, Optional

import pandas as pd
from termcolor import colored

from dataset_files.math_files.math_equivalence import is_equiv
from src.dataset_files.math_files.util import last_boxed_only_string, remove_boxed
from src.dataset_files.mmlupro_files.mmlu_pro_evaluate_from_api import (
    load_mmlu_pro,
    extract_answer as extract_answer_mmlu_pro,
)
from src.dataset_files.simple_evals.human_eval_files.human_eval.data import read_problems
from src.dataset_files.simple_evals.humaneval_eval import evaluate_functional_correctness_modified


def check_correct_answer(answer, correct_answer, dataset_name, problem_data):
    if dataset_name == "math":
        return is_equiv(answer, correct_answer)
    elif dataset_name == "mmlu-pro":
        return answer == correct_answer
    elif dataset_name == "gpqa-diamond":
        return answer == correct_answer
    elif dataset_name == "humaneval":
        # In humaneval, we don't compare to correct_answer. We just run the unit tests.
        sample = {
            "prompt": problem_data["problem"],
            "test": problem_data["test"],
            "task_id": problem_data["task_id"],
            "entry_point": problem_data["entry_point"],
        }
        passed, results = evaluate_functional_correctness_modified(sample, [answer])
        return sum(passed)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def extract_answer(solution: str, dataset_name: str, err_msg: Optional[str] = None) -> str:
    """Extract the answer from the solution."""
    if dataset_name == "math":
        # use provided extraction function
        answer = remove_boxed(last_boxed_only_string(solution))
        answer = answer.replace("**", "")
    elif dataset_name == "mmlu-pro":
        solution = solution.replace("**", "")
        answer = extract_answer_mmlu_pro(solution)
    elif dataset_name == "gpqa-diamond":
        solution = solution.replace("**", "")
        answer = extract_answer_mmlu_pro(solution)
    elif dataset_name == "humaneval":
        def find_code(completion):
            pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
            matches = pattern.findall(completion)
            extracted_answer = matches[0] if len(matches) >= 1 else completion
            extracted_answer = extracted_answer[
                               extracted_answer.find(":\n    ") + 2:
                               ]  # remove signature
            return extracted_answer

        answer = find_code(solution)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not answer:
        # Answer is None or empty string
        if isinstance(answer, str) and len(answer) == 0:
            # Is empty string, check if '\\boxed{}' is present (if present, extracted answer is empty string)
            if "\\boxed{}" in solution:
                return ""
        print(
            colored(f"\nWARNING in extract_answer, found no answer: {answer=} with {type(answer)=} ({dataset_name=}), "
                    f"and full solution (length {len(solution)}) is: \n{'-' * 30}\n{solution}\n{'-' * 30} (WARNING in extract_answer)\n"
                    f"{('     ERROR MESSAGE: ' + err_msg) if err_msg is not None else ''}", "yellow"))
        return None

    return answer


def format_multichoice_question_with_letters(question_text: str, options: list) -> tuple[str, list[str]]:
    """Format a multiple choice question, use English letters ABCDEFGHIJ in order (as needed).

    Return the formatted question and a list of the letters used.

    Example format with 4 options (enter = newline):
    {question_text}
    Options:
    A. {option1}
    B. {option2}
    C. {option3}
    D. {option4}
    """
    letters = "ABCDEFGHIJ"
    if len(options) > len(letters):
        raise ValueError(f"Max number of choices is {len(letters)}, got {len(options)}")

    question_with_options = f"{question_text}\nOptions: "
    letters_used = []
    for i, option in enumerate(options):
        question_with_options += f"\n{letters[i]}. {option}"
        letters_used.append(letters[i])
    return question_with_options, letters_used


def get_problem_info_dict(dataset_name, datapoint, err_msg):
    """Get a dictionary with the problem info."""
    # Ground truth solution for problems that don't come with a labeled ground truth solution:
    no_solution_gt = 'N/A'

    if dataset_name == "math":
        solution_gt = datapoint['solution']
        correct_answer = extract_answer(solution_gt, dataset_name, err_msg=err_msg)
        problem = datapoint['problem']

        problem_info = {
            'dataset_name': dataset_name,
            'problem': problem,
            'solution_gt': solution_gt,
            'correct_answer': correct_answer,
        }
    elif dataset_name == "mmlu-pro":
        solution_gt = no_solution_gt
        correct_answer = datapoint["answer"]
        # problem requires multichoice formatting
        question_text = datapoint["question"]
        options = datapoint["options"]
        problem, letters_used = format_multichoice_question_with_letters(question_text, options)
        # sanity check
        assert correct_answer in letters_used, f"{correct_answer=} not in {letters_used=}"

        problem_info = {
            'dataset_name': dataset_name,
            'problem': problem,
            'solution_gt': solution_gt,
            'correct_answer': correct_answer,
            # dataset-specific:
            'question_id': datapoint["question_id"],
            'src': datapoint["src"],
            'category': datapoint["category"],
        }
    elif dataset_name == "gpqa-diamond":
        # preprocessing
        choices = [
            datapoint["Correct Answer"],
            datapoint["Incorrect Answer 1"],
            datapoint["Incorrect Answer 2"],
            datapoint["Incorrect Answer 3"],
        ]
        choices = [choices[i] for i in datapoint["permutation"]]
        correct_index = choices.index(datapoint["Correct Answer"])
        # now we can define correct_answer
        solution_gt = no_solution_gt
        correct_answer = "ABCD"[correct_index]
        # problem requires multichoice formatting
        question_text = datapoint["Question"]
        options = choices
        problem, letters_used = format_multichoice_question_with_letters(question_text, options)
        # sanity check
        assert correct_answer in letters_used, f"{correct_answer=} not in {letters_used=}"

        problem_info = {
            'dataset_name': dataset_name,
            'problem': problem,
            'solution_gt': solution_gt,
            'correct_answer': correct_answer,
        }
    elif dataset_name == "humaneval":
        solution_gt = no_solution_gt
        correct_answer = datapoint['canonical_solution']
        problem = datapoint['prompt']

        problem_info = {
            'dataset_name': dataset_name,
            'problem': problem,
            'solution_gt': solution_gt,
            'correct_answer': correct_answer,
            # dataset-specific:
            'test': datapoint['test'],
            'task_id': datapoint['task_id'],
            'entry_point': datapoint['entry_point'],
        }
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    if correct_answer is None:
        raise ValueError(f"Correct answer is None for problem: {problem_info}")
    return problem_info


def load_dataset_problem_infos(dataset_name: str, math_dataset_dirpath: Optional[str]) -> list[Dict[str, str]]:
    """Load the dataset as a list of problem_info dicts."""
    if dataset_name == "math":
        if not math_dataset_dirpath:
            raise ValueError("math_test_dirpath must be filled to use the MATH dataset. Is currently empty.")
        problem_infos = []
        topics = os.listdir(math_dataset_dirpath)
        for topic in topics:
            topic_dirpath = os.path.join(math_dataset_dirpath, topic)
            problem_filenames = os.listdir(topic_dirpath)
            for problem_filename in problem_filenames:
                problem_filepath = os.path.join(topic_dirpath, problem_filename)
                with open(problem_filepath, 'r') as f:
                    datapoint = json.load(f)
                    problem_info = get_problem_info_dict(dataset_name, datapoint,
                                                         f'problem filepath: {problem_filepath}')
                    if problem_info is not None:
                        problem_infos.append(problem_info)
                    else:
                        print(colored(f'SKIPPED invalid problem at {problem_filepath}', 'red'))
    elif dataset_name == "humaneval":
        examples = read_problems()
        examples = list(examples.values())
        problem_infos = []
        for i, example in enumerate(examples):
            problem_info = get_problem_info_dict(dataset_name, example,
                                                 f'example {example["entry_point"]} with task_id: {example["task_id"]} (idx {i})')
            if problem_info is not None:
                problem_infos.append(problem_info)
            else:
                print(colored(
                    f'SKIPPED invalid problem {example["entry_point"]} with task_id: {example["task_id"]} (idx {i})',
                    'red'))
    elif dataset_name == "gpqa-diamond":
        # Note: we use the diamond variant (hardcoded in url)
        url = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
        df = pd.read_csv(url)
        examples = [row.to_dict() for _, row in df.iterrows()]
        rng = random.Random(0)
        examples = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]
        problem_infos = []
        for i, example in enumerate(examples):
            problem_info = get_problem_info_dict(dataset_name, example,
                                                 f'example with idx {i})')
            if problem_info is not None:
                problem_infos.append(problem_info)
            else:
                print(colored(f'SKIPPED invalid problem with idx {i}', 'red'))
    elif dataset_name == "mmlu-pro":
        test_df, _ = load_mmlu_pro()  # only load the test data
        examples = [example for category in test_df for example in test_df[category]]
        problem_infos = []
        for i, example in enumerate(examples):
            problem_info = get_problem_info_dict(dataset_name, example,
                                                 f'example with idx {i} and question_id {example["question_id"]} (src: {example["src"]})')
            if problem_info is not None:
                problem_infos.append(problem_info)
            else:
                print(colored(f'SKIPPED invalid problem with idx {i}', 'red'))
    else:
        raise ValueError("Expected dataset_name to be 'gsm8k' or 'math'.")

    # Random shuffle
    random.shuffle(problem_infos)

    return problem_infos
