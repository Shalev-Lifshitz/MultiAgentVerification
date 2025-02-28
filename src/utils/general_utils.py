import json
import os
from typing import Optional

from termcolor import colored

from src.utils.custom_errors import CustomJSONDecodeError


def is_solution_fully_verified(solution_vera_approvals, veras):
    """Check all verifiers have approvals for a single solution."""
    return all(
        f"{vera_model}_{vera_name}" in solution_vera_approvals
        for vera_model, vera_name in veras
    )


def check_all_problem_info_exists(problem_filepath, n_solutions, veras):
    if not os.path.exists(problem_filepath):
        return False
    problem_data = load_problem_data(problem_filepath)

    # Check if all required keys exist
    if 'solutions' not in problem_data:
        return False
    if 'solution_gt' not in problem_data:
        return False
    if 'all_solution_vera_approvals' not in problem_data:
        return False

    # Check if there are enough solutions and if they are fully verified
    existing_solutions = problem_data.get('solutions', [])
    enough_solutions = len(existing_solutions) >= n_solutions
    enough_solutions_fully_verified = all(is_solution_fully_verified(solution_vera_approvals, veras)
                                          for solution_vera_approvals in problem_data['all_solution_vera_approvals'][:n_solutions])

    return enough_solutions and enough_solutions_fully_verified


def find_missing_problems(problems, solutions_dirpath, n_outputs, veras):
    problems_to_process = []
    for i in range(len(problems)):
        problem_filepath = os.path.join(solutions_dirpath, f"problem_{i}.json")
        if not os.path.exists(problem_filepath):
            problems_to_process.append(i)
        else:
            if not check_all_problem_info_exists(problem_filepath, n_outputs, veras):
                problems_to_process.append(i)
    return problems_to_process


def get_run_name(dataset_name: str, gen_model: str, prefix: Optional[str] = None) -> str:
    run_name = f"{dataset_name}_{gen_model}"
    if prefix:
        run_name = f"{prefix}_{run_name}"
    return run_name


def save_update_problem_data(problem_data, problem_filepath):
    with open(problem_filepath, 'w') as f:
        json.dump(problem_data, f, indent=4)


def load_problem_data(problem_filepath):
    if not os.path.exists(problem_filepath):
        raise ValueError(f"Problem file not found: {problem_filepath}")

    problem_data = None
    try:
        with open(problem_filepath, 'r') as f:
            problem_data = json.load(f)
    except Exception as e:
        if "JSONDecodeError" in type(e).__name__:
            file_size = os.path.getsize(problem_filepath)
            msg = f"Encountered JSONDecodeError when loading problem data (file size: {file_size}) from {problem_filepath}"
            raise CustomJSONDecodeError(msg, file_size)
        else:
            print(colored(f"\nUnexpected error '{type(e).__name__}' when loading problem data: {problem_filepath}\n", 'red'))
            raise e

    assert problem_data is not None, f"Problem data is None: {problem_filepath}"
    return problem_data


