import time

from src.utils.api_utils import (
    query_openai,
    query_gemini,
    raw_to_api_messages,
)
from src.prompts.gen_prompts import get_gen_prompt


def generate_answer_openai(problem: str, model: str, temperature: float, dataset_name: str) -> str:
    """Generate an answer with reasoning using an OpenAI model."""
    user_prompt = get_gen_prompt(dataset_name, problem)
    raw_messages = [
        [False, user_prompt],
    ]
    messages = raw_to_api_messages(raw_messages, "openai")
    response = query_openai(model, messages, temperature, "generate_answer_openai")
    return response.strip()


def generate_answer_gemini(problem: str, model: str, temperature: float, dataset_name: str) -> str:
    """Generate an answer with reasoning using a Gemini model."""
    user_prompt = get_gen_prompt(dataset_name, problem)
    raw_messages = [
        [False, user_prompt]
    ]
    messages = raw_to_api_messages(raw_messages, "gemini")
    response = query_gemini(model, messages, temperature, "generate_answer_gemini")
    return response.strip()


def generate_answer(problem: str, model: str, temperature: float, dataset_name: str) -> tuple:
    """Generate an answer with reasoning using the specified model."""
    start_time = time.time()
    if model.startswith("gpt"):
        response = generate_answer_openai(problem, model, temperature, dataset_name)
    elif model.startswith("gemini"):
        response = generate_answer_gemini(problem, model, temperature, dataset_name)
    else:
        raise ValueError(f"Unsupported model: {model}")
    end_time = time.time()
    return response, end_time - start_time
