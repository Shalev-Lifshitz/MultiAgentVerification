VERA_ANSWER_SYMBOL = "FINAL VERIFICATION ANSWER:"

# For verifiers other than direct approval, we ask a follow up message since it is sometimes unclear what the verifier decided
VERA_ASK_FOR_APPROVAL_ONLY_PROMPT = f"To clarify, based on the above analysis, reply with ONLY '{VERA_ANSWER_SYMBOL}True' or ONLY '{VERA_ANSWER_SYMBOL}False'. Do not include any other text in your response."


def is_not_direct_approval(vera_name: str) -> bool:
    return "direct" not in vera_name


def get_vera_prompt(dataset_name, vera_name, question, solution):
    # system string should be a single line (no newlines)
    system_str_math = (
        "You are a critical verifier tasked with evaluating mathematical problem-solving. "
        "You will be presented with a question and a proposed solution. "
        "Your job is to carefully go over and analyze the solution. Follow the instructions."
    )
    system_str_code = (
        "You are a critical verifier tasked with evaluating code implementations. "
        "You will be presented with a prompt and a code implementation. "
        "Your job is to carefully go over and analyze the code. Follow the instructions."
    )
    system_str_multiple_choice = (
        "You are a critical verifier tasked with evaluating multiple-choice question-answering. "
        "You will be presented with a question, the multiple-choice options, and a proposed solution. "
        "Your job is to carefully go over and analyze the solution. Follow the instructions."
    )
    if dataset_name in ["math"]:
        system_str = system_str_math
    elif dataset_name in ["mmlu-pro", "gpqa-diamond"]:
        system_str = system_str_multiple_choice
    elif dataset_name == "humaneval":
        system_str = system_str_code
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    prefix = f"""{system_str}\n\n
    QUESTION:
    {question}\n\n
    PROPOSED SOLUTION:
    {solution}\n\n"""

    vera_names_to_prompts = {
        "math_steps": (
            f"{prefix}"
            "INSTRUCTIONS: \n"
            f"Go over each step in the proposed solution and check whether it is mathematically correct. Think out load. "
            f"If you reach a step that is incorrect, stop and reply '{VERA_ANSWER_SYMBOL}False'."
            f"If you get to the end of all the steps and each step was correct, reply '{VERA_ANSWER_SYMBOL}True'."
        ),
        "logic_steps": (
            f"{prefix}"
            "INSTRUCTIONS: \n"
            f"Go over each step in the proposed solution and check whether it is logically sound. Think out load. "
            f"If you reach a step that is not logically sound, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
            f"If you get to the end of all the steps and each step was logically sound, reply '{VERA_ANSWER_SYMBOL}True'."
        ),
        "facts_steps": (
            f"{prefix}"
            "INSTRUCTIONS: \n"
            f"Go over each step in the proposed solution and check whether the facts presented are correct. Think out load. "
            f"If you reach a step with incorrect facts, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
            f"If you get to the end of all the steps and each step had correct facts, reply '{VERA_ANSWER_SYMBOL}True'."
        ),
        "units_steps": (
            f"{prefix}"
            "INSTRUCTIONS: \n"
            f"Check if the units are handled correctly in each step of the solution. Think out loud. "
            f"If you find any issues with the units, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
            f"If all units are handled correctly, reply '{VERA_ANSWER_SYMBOL}True'."
        ),
        "general_direct": (
            f"{prefix}"
            f"INSTRUCTIONS: \n"
            f"Is this solution correct for the given question? "
            f"Respond with ONLY '{VERA_ANSWER_SYMBOL}True' or ONLY '{VERA_ANSWER_SYMBOL}False'. Do not provide any explanation or additional text."
        ),
        "general_summarize": (
            f"{prefix}"
            "INSTRUCTIONS: \n"
            f"Summarize the solution in your own words, explore anything you think may be incorrect. Think out load. "
            f"If you find something that's incorrect, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
            f"If you've gone over the solution and everything seems correct, reply '{VERA_ANSWER_SYMBOL}True'."
        ),
        "general_diff": (
            f"{prefix}"
            "INSTRUCTIONS: \n"
            f"Explain the solution in a different way than it was presented. "
            "Try to find any flaws in the solution. Think out load. "
            f"If you find something that's incorrect, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
            f"If you've gone over the solution and everything seems correct, reply '{VERA_ANSWER_SYMBOL}True'."
        ),
        "general_edge": (
            f"{prefix}"
            "INSTRUCTIONS: \n"
            f"Check if the solution handles edge cases and boundary conditions, test extreme values or special cases. Think out loud. "
            f"If any boundary conditions or edge cases fail, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
            f"If all boundary conditions and edge cases are handled correctly, reply '{VERA_ANSWER_SYMBOL}True'."
        ),
        "general_mistakes": (
            f"{prefix}"
            "INSTRUCTIONS: \n"
            f"Check if the solution has any common mistakes, calculation errors, or misconceptions that typically found in this type of problem. Think out loud. "
            f"If you find any common mistakes, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
            f"If no common mistakes are found, reply '{VERA_ANSWER_SYMBOL}True'."
        ),
        "general_domain": (
            f"{prefix}"
            "INSTRUCTIONS: \n"
            f"Check if the solution correctly applies relevant domain-knowledge, established theories, and standard practices for this type of problem. Think out loud. "
            f"If any domain knowledge is misapplied or violated, stop and reply '{VERA_ANSWER_SYMBOL}False'. "
            f"If all domain-specific knowledge is correctly applied, reply '{VERA_ANSWER_SYMBOL}True'."
        ),
    }
    return vera_names_to_prompts[vera_name]
