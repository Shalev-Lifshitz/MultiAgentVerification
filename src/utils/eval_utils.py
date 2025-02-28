from collections import Counter

from termcolor import colored

from src.utils.dataset_utils import check_correct_answer


def get_extracted_answers_null_to_empty_str(problem_data):
    extracted_answers = []
    for answer in problem_data['extracted_answers']:
        if answer is None:
            extracted_answers.append("")
        else:
            extracted_answers.append(answer)
    return extracted_answers


def get_n_answers_and_approvals(problem_data, n_solutions):
    sampled_indices = list(range(n_solutions))
    extracted_answers = get_extracted_answers_null_to_empty_str(problem_data)
    extracted_answers = [extracted_answers[i] for i in sampled_indices]
    solution_vera_approvals = [problem_data['all_solution_vera_approvals'][i] for i in sampled_indices]
    return extracted_answers, solution_vera_approvals, sampled_indices


def compute_aggregated_verification_score(vera_approvals_for_solution, veras):
    num_positive_approvals = 0
    for vera_model, vera_name in veras:
        approval_key = f"{vera_model}_{vera_name}"
        if approval_key in vera_approvals_for_solution:
            num_positive_approvals += vera_approvals_for_solution[approval_key]
        else:
            raise ValueError(f"Missing approval_key {approval_key} in solution_vera_approvals")
    aggregated_verification_score = num_positive_approvals / len(veras)
    return aggregated_verification_score


def evaluate_problem_bon_mav(dataset_name, problem_data, n_solutions, veras):
    correct_answer = problem_data['correct_answer']
    extracted_answers, solution_vera_approvals, _ = get_n_answers_and_approvals(problem_data, n_solutions)

    # Get the best-rated answer by the verifiers
    agg_score_key = lambda i: compute_aggregated_verification_score(solution_vera_approvals[i], veras)
    best_solution_index = max(range(len(solution_vera_approvals)), key=agg_score_key)
    best_extracted_answer = extracted_answers[best_solution_index]

    if best_extracted_answer is None:
        # No answer can be extracted from this solution, but it was selected as the best solution, mark as incorrect.
        return False

    is_correct = check_correct_answer(best_extracted_answer, correct_answer, dataset_name, problem_data)
    assert isinstance(is_correct, bool) or is_correct in [0, 1]
    return int(is_correct)


def get_self_consistency_answer(extracted_answers: list) -> str:
    if not extracted_answers:
        raise ValueError("No extracted answers to get self-consistency answer from.")
    vote_counts = Counter(extracted_answers)
    return max(vote_counts, key=vote_counts.get)


def evaluate_problem_self_consistency(dataset_name, problem_data, n_solutions):
    if n_solutions < 3:
        # Self-consistency is not possible with less than 3 solutions
        raise ValueError(colored(f"Self-consistency is not possible with less than 3 solutions, got {n_solutions=}.", 'yellow'))

    correct_answer = problem_data['correct_answer']
    extracted_answers, _, _ = get_n_answers_and_approvals(problem_data, n_solutions)
    self_consistency_answer = get_self_consistency_answer(extracted_answers)

    if self_consistency_answer is None:
        # The selected answer is null (i.e., most sampled solutions were not able to have an answer successfully extracted), mark as incorrect.
        return False

    is_correct = check_correct_answer(self_consistency_answer, correct_answer, dataset_name, problem_data)
    assert isinstance(is_correct, bool) or is_correct in [0, 1]
    return int(is_correct)


def evaluate_problem_pass_at_1(dataset_name, problem_data):
    correct_answer = problem_data['correct_answer']
    extracted_answers, _, _ = get_n_answers_and_approvals(problem_data, 1)
    pass_at_1_answer = extracted_answers[0]

    if pass_at_1_answer is None:
        # The selected answer is null (i.e., the solution was not able to have an answer successfully extracted), mark as incorrect.
        return False

    is_correct = check_correct_answer(pass_at_1_answer, correct_answer, dataset_name, problem_data)
    assert isinstance(is_correct, bool) or is_correct in [0, 1]
    return int(is_correct)