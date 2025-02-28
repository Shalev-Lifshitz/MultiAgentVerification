import multiprocessing
import os
import random
import numpy as np
import argparse

from tqdm import tqdm
from sys import stdout

from src.utils.dataset_utils import load_dataset_problem_infos
from src.utils.eval_utils import evaluate_problem_bon_mav, evaluate_problem_pass_at_1, evaluate_problem_self_consistency
from src.utils.plotting import gen_solution_scaling_plot
from src.process_problem import process_problem_with_retry
from src.utils.general_utils import find_missing_problems, get_run_name, load_problem_data

PROJECT_DIRPATH = "./"
MATH_DATASET_DIRPATH_TST = "TODO: MUST FILL THIS IN TO USE MATH"  # TODO: this is necessary to use the math dataset.

RUNS_DIRPATH = os.path.join(PROJECT_DIRPATH, "runs")
os.makedirs(RUNS_DIRPATH, exist_ok=True)

assert os.getenv("OPENAI_API_KEY") not in [None, ""], "Please set the OPENAI_API_KEY environment variable."
assert os.getenv("GEMINI_API_KEY") not in [None, ""], "Please set the GEMINI_API_KEY environment variable."


def load_domain_specific_verifiers(dataset_name):
    if dataset_name == "math":
        veras = [
            ("gpt-4o-mini-2024-07-18", "units_steps"),
            ("gpt-4o-mini-2024-07-18", "general_summarize"),
            ("gpt-4o-mini-2024-07-18", "general_edge"),
            ("gpt-4o-mini-2024-07-18", "general_mistakes"),
            ("gpt-4o-mini-2024-07-18", "general_domain"),
            ("gemini-1.5-flash-001", "general_edge"),
        ]
    elif dataset_name == "mmlu-pro":
        veras = [
            ("gpt-4o-mini-2024-07-18", "math_steps"),
            ("gpt-4o-mini-2024-07-18", "logic_steps"),
            ("gpt-4o-mini-2024-07-18", "general_diff"),
            ("gpt-4o-mini-2024-07-18", "general_edge"),
            ("gpt-4o-mini-2024-07-18", "general_mistakes"),
            ("gpt-4o-mini-2024-07-18", "general_domain"),
            ("gemini-1.5-flash-001", "units_steps"),
            ("gemini-1.5-flash-001", "general_mistakes"),
        ]
    elif dataset_name == "gpqa-diamond":
        veras = [
            ("gpt-4o-mini-2024-07-18", "math_steps"),
            ("gpt-4o-mini-2024-07-18", "logic_steps"),
            ("gpt-4o-mini-2024-07-18", "units_steps"),
            ("gpt-4o-mini-2024-07-18", "general_diff"),
            ("gemini-1.5-flash-001", "units_steps"),
            ("gemini-1.5-flash-001", "general_diff"),
            ("gemini-1.5-flash-001", "general_mistakes"),
        ]
    elif dataset_name == "humaneval":
        veras = [
            ("gpt-4o-mini-2024-07-18", "math_steps"),
            ("gpt-4o-mini-2024-07-18", "logic_steps"),
            ("gpt-4o-mini-2024-07-18", "facts_steps"),
            ("gpt-4o-mini-2024-07-18", "units_steps"),
            ("gpt-4o-mini-2024-07-18", "general_direct"),
            ("gpt-4o-mini-2024-07-18", "general_diff"),
            ("gpt-4o-mini-2024-07-18", "general_edge"),
            ("gpt-4o-mini-2024-07-18", "general_domain"),
            ("gemini-1.5-flash-001", "logic_steps"),
            ("gemini-1.5-flash-001", "units_steps"),
            ("gemini-1.5-flash-001", "general_direct"),
            ("gemini-1.5-flash-001", "general_summarize"),
            ("gemini-1.5-flash-001", "general_diff"),
            ("gemini-1.5-flash-001", "general_domain"),
        ]
    else:
        raise ValueError(f"Domain-specific verifiers not implemented for dataset {dataset_name}.")
    return veras


def generate_data(problem_infos, run_dirpath, gen_model, veras, gen_temp, vera_temp, n_solutions):
    solutions_dirpath = os.path.join(run_dirpath, "solutions")
    while True:
        missing_problems_indices = find_missing_problems(problem_infos, solutions_dirpath, n_solutions, veras)
        if not missing_problems_indices:
            break

        print("\n" + "-" * 50 + f"\nPROCESSING {len(missing_problems_indices)} REMAINING PROBLEMS...\n" + "-" * 50 + "\n")
        args_list = [(problem_infos[i], i, gen_model, veras, gen_temp, vera_temp, n_solutions, solutions_dirpath)
                     for i in missing_problems_indices]

        print(f"Number of CPUs available: {multiprocessing.cpu_count()}")
        with multiprocessing.Pool() as pool:
            with tqdm(total=len(args_list), desc="Processing problems", file=stdout) as pbar:
                for result in pool.imap_unordered(process_problem_with_retry, args_list):
                    if result is not None:
                        pbar.update()


def evaluate_data(dataset_name, run_dirpath, veras, n_problems, n_solutions, include_self_cons=False, include_bon_mav=True):
    solutions_dirpath = os.path.join(run_dirpath, "solutions")
    total_correct_bon_mav = [0] * n_solutions if include_bon_mav else None
    total_correct_self_cons = [0] * n_solutions if include_self_cons else None
    total_correct_pass_at_1 = 0

    for problem_idx in tqdm(range(n_problems), desc=f"Evaluating {n_problems} problems..."):
        problem_filepath = os.path.join(solutions_dirpath, f"problem_{problem_idx}.json")
        problem_data = load_problem_data(problem_filepath)

        for n in range(1, n_solutions + 1):
            if include_bon_mav:
                is_correct_bon_mav = evaluate_problem_bon_mav(dataset_name, problem_data, n, veras)
                total_correct_bon_mav[n - 1] += is_correct_bon_mav

            if include_self_cons and n >= 3:
                is_correct_self_cons = evaluate_problem_self_consistency(dataset_name, problem_data, n)
                total_correct_self_cons[n - 1] += is_correct_self_cons
            elif include_self_cons:
                total_correct_self_cons[n - 1] = None

        is_correct_pass_at_1 = evaluate_problem_pass_at_1(dataset_name, problem_data)
        total_correct_pass_at_1 += is_correct_pass_at_1

    bon_mav_accuracies = [(correct / n_problems) * 100 for correct in total_correct_bon_mav] if include_bon_mav else None
    self_cons_accuracies = [None if correct is None else (correct / n_problems) * 100 for correct in total_correct_self_cons] if include_self_cons else None
    pass_at_1_accuracy = (total_correct_pass_at_1 / n_problems) * 100

    plot_save_filepath = os.path.join(run_dirpath, "solution_scaling_plot.png")
    gen_solution_scaling_plot(dataset_name, bon_mav_accuracies, self_cons_accuracies, pass_at_1_accuracy, plot_save_filepath)

    print("\n" + "-" * 50 + "\nRESULTS\n" + "-" * 50)
    if include_bon_mav:
        print(f"BoN-MAV@{n_solutions}: {bon_mav_accuracies[-1]:.2f}%")
    if include_self_cons:
        print(f"Self-consistency@{n_solutions}: {self_cons_accuracies[-1]:.2f}%")
    print(f"Pass@1: {pass_at_1_accuracy:.2f}%")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Verification")
    parser.add_argument("--self-cons", action="store_true", help="Include self-consistency baseline")
    parser.add_argument("--bon-mav", action="store_true", help="Include BoN-MAV results")
    parser.add_argument("--use-example-data", action="store_true", help="Use pre-generated example data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset", type=str, default="math", help="Dataset to use (math, mmlu-pro, gpqa-diamond, humaneval)")
    parser.add_argument("--gen-model", type=str, default="gemini-1.5-flash-001", help="Generator model")
    parser.add_argument("--gen-temp", type=float, default=0.7, help="Generator temperature")
    parser.add_argument("--vera-temp", type=float, default=0.0, help="Verifier temperature")
    parser.add_argument("--n-problems", type=int, default=300, help="Number of problems")
    parser.add_argument("--n-solutions", type=int, default=16, help="Number of candidate outputs per problem")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load the domain-specific verifier set for this dataset
    veras = load_domain_specific_verifiers(args.dataset)

    if args.use_example_data:
        run_name = 'example_math_gemini-1.5-flash'
        run_dirpath = os.path.join(RUNS_DIRPATH, run_name)
        if not os.path.exists(run_dirpath):
            print(f"Error: Specified example data directory {run_dirpath} does not exist.")
            return
        print(f"Using pre-generated example data from: {run_dirpath}")
    else:
        # Load the dataset as a list of problem_info dicts
        problem_infos = load_dataset_problem_infos(args.dataset, math_dataset_dirpath=MATH_DATASET_DIRPATH_TST)
        problem_infos = problem_infos[:args.n_problems]  # already comes shuffled from load_dataset_problem_infos

        # Create run directory
        run_name = get_run_name(args.dataset, args.gen_model)
        run_dirpath = os.path.join(RUNS_DIRPATH, run_name)
        os.makedirs(os.path.join(run_dirpath, "solutions"), exist_ok=True)

        # Generate data and evaluate
        generate_data(problem_infos, run_dirpath, args.gen_model, veras, args.gen_temp, args.vera_temp, args.n_solutions)

    # Evaluate the generated data (or example data)
    evaluate_data(
        args.dataset,
        run_dirpath,
        veras,
        args.n_problems,
        args.n_solutions,
        include_self_cons=args.self_cons,
        include_bon_mav=args.bon_mav
    )


if __name__ == "__main__":
    main()
