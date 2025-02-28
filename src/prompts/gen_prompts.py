def format_example_custom_mmlu_pro(question, options):
    example = "{}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "\n{}. {}".format(choice_map[i], opt)
    return example


def get_gen_prompt(dataset_name, problem):
    # MATH
    gen_prompt_math = (
        "You are a helpful assistant skilled in math problem-solving. "
        "Always end your solution with the final numerical answer in latex, using '\\boxed{<answer>}'. "
        "If there is no solution, reply with an empty boxed '\\boxed{}'."
        "\nPlease solve the following math problem step by step:"
        f"\n\nQUESTION: {problem}"
        "\n\nProvide your detailed solution below:"
    )

    # MULTIPLE-CHOICE (note: problem already contains the options)
    if dataset_name in ["mmlu-pro", "gpqa-diamond"]:
        assert "options" in problem.lower(), f"Problem does not contain 'options', something is probably wrong. Promblem: \n{'-' * 30}\n{problem}\n{'-' * 30}"
    multichoice_template = """
    Answer the following multiple choice question. Think step by step before answering, and then output the answer in the format of \"The answer is (X)\" at the end, where X is the LETTER of the correct answer.

    QUESTION:
    {problem}

    Think step by step, then end with EXACTLY \"The answer is (X)\", where X is the LETTER of the correct answer. Do not include the answer text itself, only the letter.
    """
    gen_prompt_mmlu_pro = multichoice_template.format(problem=problem)
    gen_prompt_gpqa_diamond = multichoice_template.format(problem=problem)

    # CODE
    gen_prompt_humaneval = (
        "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function."
        f"\n{problem}"
    )

    if dataset_name == "math":
        return gen_prompt_math
    elif dataset_name == "mmlu-pro":
        return gen_prompt_mmlu_pro
    elif dataset_name == "gpqa-diamond":
        return gen_prompt_gpqa_diamond
    elif dataset_name == "humaneval":
        return gen_prompt_humaneval
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

