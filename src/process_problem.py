import datetime
import os
import sys
import time
import traceback

from src.utils.custom_errors import CustomRateLimitError
from src.utils.dataset_utils import extract_answer
from src.utils.gen_utils import generate_answer
from src.utils.general_utils import save_update_problem_data, load_problem_data, check_all_problem_info_exists
from src.utils.vera_utils import verify_answer

# Constants
MAX_RETRIES = 120
RETRY_DELAY = 5


def process_problem(args):
    (problem_info, problem_idx,
     gen_model, veras, gen_temp, vera_temp,
     n_solutions, solutions_dirpath) = args

    assert 'dataset_name' in problem_info
    assert 'problem' in problem_info
    assert 'solution_gt' in problem_info
    assert 'correct_answer' in problem_info

    problem_filepath = os.path.join(solutions_dirpath, f"problem_{problem_idx}.json")
    if check_all_problem_info_exists(problem_filepath, n_solutions, veras):
        return 0

    if os.path.exists(problem_filepath):
        existing_problem_data = load_problem_data(problem_filepath)
        for key in problem_info:
            assert existing_problem_data[key] == problem_info[key], \
                f"{key} mismatch in problem {problem_idx}: \n\n{existing_problem_data[key]} \n\n!= \n\n{problem_info[key]}"

    # Define default problem file structure
    default_problem_data = {key: problem_info[key] for key in problem_info}
    default_problem_data.update({
        "solutions": [],
        "extracted_answers": [],
        "all_solution_vera_approvals": [],  # Each element is a dict of approvals for one solution
        "all_solution_vera_responses": [],  # Same as all_solution_vera_approvals, but the whole response
    })
    if not os.path.exists(problem_filepath):
        save_update_problem_data(default_problem_data, problem_filepath)

    # Loop over keys in default_problem_data and add any missing keys / initial values to problem_data.
    # Do not override any data.
    def add_init_problem_data_recursive(prob_data, default_data):
        for k, init_value in default_data.items():
            if k not in prob_data:
                prob_data[k] = init_value
            elif isinstance(init_value, dict):
                add_init_problem_data_recursive(prob_data[k], init_value)

    problem_data = load_problem_data(problem_filepath)
    add_init_problem_data_recursive(problem_data, default_problem_data)
    save_update_problem_data(problem_data, problem_filepath)

    # Error message with all the relevant problem info
    err_msg_for_extract_answer = f"problem_{problem_idx} ({problem_data['dataset_name']}) at {problem_filepath}"

    # Generate new solutions if needed
    num_solutions_initial = len(problem_data['solutions'])
    new_solutions_needed = n_solutions - num_solutions_initial
    for i in range(new_solutions_needed):
        solution, gen_time = generate_answer(problem_data['problem'], gen_model, gen_temp, problem_data['dataset_name'])
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{datetime_str} GEN] Generated solution {num_solutions_initial + i + 1}/{n_solutions} for problem {problem_idx} ({gen_time:.2f} secs)", file=sys.stderr, flush=True)
        problem_data['solutions'].append(solution)
        problem_data['extracted_answers'].append(extract_answer(solution, problem_data['dataset_name'], err_msg=err_msg_for_extract_answer))
        problem_data['all_solution_vera_approvals'].append({})  # Initialize empty approvals for new solution
        problem_data['all_solution_vera_responses'].append({})  # Initialize empty approvals for new solution
        save_update_problem_data(problem_data, problem_filepath)

    # Produce approvals for all verifiers
    for solution_index, solution in enumerate(problem_data['solutions'][:n_solutions]):
        for vera_model, vera_name in veras:
            approval_key = f"{vera_model}_{vera_name}"
            print_str = f"{approval_key}={{bool_result}} for solution {solution_index + 1}/{n_solutions} in problem {problem_idx}: {problem_filepath}"
            if approval_key not in problem_data['all_solution_vera_approvals'][solution_index]:
                # This vera approval has not yet been generated
                pre_verify_time = time.time()
                bool_result, vera_response, vera_time = verify_answer(problem_data['dataset_name'],
                                                                      problem_data['problem'], solution,
                                                                      vera_name, vera_model, vera_temp)
                verify_time_taken = time.time() - pre_verify_time
                problem_data['all_solution_vera_approvals'][solution_index][approval_key] = bool_result
                problem_data['all_solution_vera_responses'][solution_index][approval_key] = vera_response
                save_update_problem_data(problem_data, problem_filepath)
                # Print approval
                datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{datetime_str} VERA] Generated vera approval {print_str.format(bool_result=bool_result)} ({verify_time_taken:.2f} secs)", file=sys.stderr, flush=True)

    return 0


def handle_general_exception(e, problem_idx, attempt, max_retries, retry_delay):
    datetime_str = time.strftime("%Y-%m-%d %H:%M:%S")
    if attempt < max_retries - 1:
        error_message = (f"[{datetime_str} ERROR] Error in problem {problem_idx}, "
                         f"attempt {attempt + 1}/{max_retries}: {type(e).__name__}: {str(e)}. "
                         f"RETRYING in {retry_delay} seconds...")
        print(error_message, file=sys.stderr, flush=True)
        print(f"\nError is: {e}", file=sys.stderr, flush=True)
        print("Traceback is:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        time.sleep(retry_delay)
    else:
        error_message = (f"\n[{datetime_str} ERROR] Max retries reached for problem {problem_idx}. "
                         f"Error: {type(e).__name__}: {str(e)}. SKIPPING this problem.\n")
        print(error_message, file=sys.stderr, flush=True)
        print(f"\tError is: {e}", file=sys.stderr, flush=True)
        print("Traceback is:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return None


def process_problem_with_retry(args, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    problem_idx = args[1]
    for attempt in range(max_retries):
        try:
            return process_problem(args)
        except AssertionError:
            raise  # This is a bug, so we want to raise it
        except CustomRateLimitError as e:
            # Rate limit exceeded, retry after delay or skip if max retries reached
            datetime_str = time.strftime("%Y-%m-%d %H:%M:%S")
            if attempt < max_retries - 1:
                error_message = (
                    f"[{datetime_str} ERROR] Rate limit exceeded for {e.model} in {e.function} for problem {problem_idx}, "
                    f"attempt {attempt + 1}/{max_retries}. "
                    f"RETRYING in {retry_delay} seconds..."
                )
                if e.original_message:
                    error_message += f"\n\tError is: {e.original_message}"
                print(error_message, file=sys.stderr, flush=True)
                time.sleep(retry_delay)
            else:
                error_message = (f"\n[{datetime_str} ERROR] Max retries reached for problem {problem_idx} "
                                 f"due to rate limiting. SKIPPING this problem.\n")
                if e.original_message:
                    error_message += f"\n\tError is: {e.original_message}"
                print(error_message, file=sys.stderr, flush=True)
                return None
        except Exception as e:
            return handle_general_exception(e, problem_idx, attempt, max_retries, retry_delay)
