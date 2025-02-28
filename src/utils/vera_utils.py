import time
from typing import Tuple, Union

from termcolor import colored

from src.utils.api_utils import (
    query_openai,
    query_gemini,
    raw_to_api_messages,
)
from src.prompts.vera_prompts import (
    get_vera_prompt,
    VERA_ASK_FOR_APPROVAL_ONLY_PROMPT,
    VERA_ANSWER_SYMBOL,
    is_not_direct_approval
)


def extract_verifier_approval(verifier_response: str) -> bool:
    """Extract the verifier's approval from the response."""
    # Get the last answer
    vera_answer_symbol = VERA_ANSWER_SYMBOL.lower()
    last_index = verifier_response.lower().rfind(vera_answer_symbol)
    answer = verifier_response[last_index + len(vera_answer_symbol):].strip() if last_index != -1 else None
    
    if not answer:
        print(colored(f"WARNING in extract_verifier_approval: {answer=} with {type(answer)=}, "
                      f"and full verifier_response (length {len(verifier_response)}): "
                      f"\n{'-' * 30}\n{verifier_response}\n{'-' * 30} (WARNING in extract_verifier_approval)\n", "yellow"))
        return False
    
    answer = answer.replace("*", "")  # Remove any asterisks (bolding)
    answer = answer.strip().lower()
    if answer == "true" or answer == "true.":
        return True
    elif answer == "false" or answer == "false.":
        return False
    else:
        # Check if 'true' or 'false' is in the first word
        print(colored(f"NOTICE in extract_verifier_approval: {answer=} with {type(answer)=} is not 'true' or 'false', "
                      f"checking if the FIRST WORK contains 'true' or 'false'...", "magenta"))
        first_word = answer.split()[0]
        if "true" in first_word:
            print(colored(f"\tSuccess. Found 'true' in first_word.lower(): {first_word.lower()}", "magenta"))
            return True
        elif "false" in first_word:
            print(colored(f"\tSuccess. Found 'false' in first_word.lower(): {first_word.lower()}", "magenta"))
            return False
        else:
            print(colored(f"WARNING in extract_verifier_approval: {answer=} with {type(answer)=} is not 'true' or 'false', "
                          f"AND first word does not contain 'true' or 'false. Full verifier_response: "
                          f"\n{'-' * 30}\n{verifier_response}\n{'-' * 30} (WARNING in extract_verifier_approval)\n", "yellow"))
            return False
    

def verify_answer_openai(model: str, temperature: float, user_prompt: str, vera_name: str) -> Tuple[bool, Union[str, list[str]]]:
    """Get the verifier approval for the answer using an OpenAI model."""
    raw_messages = [
        [False, user_prompt]
    ]
    messages = raw_to_api_messages(raw_messages, "openai")
    response = query_openai(model, messages, temperature, "verify_answer_openai")
    
    if is_not_direct_approval(vera_name):
        # Response may not be very clear, so ask for approval only
        raw_messages = raw_messages + [
            [True, response],
            [False, VERA_ASK_FOR_APPROVAL_ONLY_PROMPT]
        ]
        messages = raw_to_api_messages(raw_messages, "openai")
        approval_response = query_openai(model, messages, temperature, "verify_answer_openai")
        approval_bool = extract_verifier_approval(approval_response)
        return approval_bool, [response, approval_response]
    else:
        approval_bool = extract_verifier_approval(response)
        return approval_bool, response


def verify_answer_gemini(model: str, temperature: float, user_prompt: str, vera_name: str) -> Tuple[bool, Union[str, list[str]]]:
    """Get the verifier approval for the answer using a Gemini model."""
    raw_messages = [
        [False, user_prompt]
    ]
    messages = raw_to_api_messages(raw_messages, "gemini")
    response = query_gemini(model, messages, temperature, "verify_answer_gemini")
    
    if is_not_direct_approval(vera_name):
        # Response may not be very clear, so ask for approval only
        raw_messages = raw_messages + [
            [True, response],
            [False, VERA_ASK_FOR_APPROVAL_ONLY_PROMPT]
        ]
        messages = raw_to_api_messages(raw_messages, "gemini")
        approval_response = query_gemini(model, messages, temperature, "verify_answer_gemini")
        approval_bool = extract_verifier_approval(approval_response)
        return approval_bool, [response, approval_response]
    else:
        approval_bool = extract_verifier_approval(response)
        return approval_bool, response


def verify_answer(dataset_name: str, question: str, solution: str, vera_name: str, model: str, temperature: float) -> Tuple[bool, Union[str, list[str]], float]:
    """Get the verifier approval for the answer using a specified model."""
    start_time = time.time()
    
    user_prompt = get_vera_prompt(dataset_name, vera_name, question, solution)
    
    if model.startswith("gpt"):
        approval_bool, vera_response = verify_answer_openai(model, temperature, user_prompt, vera_name)
    elif model.startswith("gemini"):
        approval_bool, vera_response = verify_answer_gemini(model, temperature, user_prompt, vera_name)
    else:
        raise ValueError(f"Unsupported model: {model}")
    
    end_time = time.time()
    return approval_bool, vera_response, end_time - start_time
