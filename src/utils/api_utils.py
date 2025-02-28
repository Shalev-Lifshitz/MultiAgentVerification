import os

import google.generativeai as genai
import openai
from openai import OpenAI
from termcolor import colored

from src.utils.custom_errors import CustomRateLimitError

# Get API keys from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


MAX_TOKENS = 4096
GEMINI_SAFETY_SETTINGS = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                          {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                          {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                          {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]


def raw_to_api_messages(raw_messages: list[list], api_name: str) -> list[dict]:
    """
    Takes a list of messages where each message is list [is_system: bool, text: str] and returns 
    a list of messages formatted for the specified API.
    """
    if api_name not in ["openai", "gemini"]:
        raise ValueError(f"Unsupported API: {api_name}")

    role_str = "role"
    user_str = "user"
    openai_system_str = "system"
    openai_content_str = "content"
    gemini_system_str = "model"
    gemini_content_str = "parts"

    api_messages = []
    for raw_message in raw_messages:
        assert len(raw_message) == 2
        assert isinstance(raw_message[0], bool)
        assert isinstance(raw_message[1], str)
        is_system = raw_message[0]
        text = raw_message[1]

        if is_system:
            if api_name == "openai":
                api_messages.append({role_str: openai_system_str, openai_content_str: text})
            elif api_name == "gemini":
                api_messages.append({role_str: gemini_system_str, gemini_content_str: text})
        else:
            if api_name == "openai":
                api_messages.append({role_str: user_str, openai_content_str: text})
            elif api_name == "gemini":
                api_messages.append({role_str: user_str, gemini_content_str: text})

    return api_messages


def query_openai(model: str, messages: list[dict], temperature: float, 
                 func_name_for_error: str) -> str:
    try:
        client = OpenAI()
        api_response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_TOKENS
        )
        response = api_response.choices[0].message.content
        return response
    except openai.RateLimitError:
        raise CustomRateLimitError(model, func_name_for_error)
    except Exception as e:
        raise


def query_gemini_subcall(model: str, messages: list[dict], temperature: float) -> str:
    model_instance = genai.GenerativeModel(
        model_name=model,
        generation_config={
            'temperature': temperature,
        },
        safety_settings=GEMINI_SAFETY_SETTINGS,
    )
    history = messages[:-1]
    prompt = messages[-1]
    chat = model_instance.start_chat(history=history)
    api_response = chat.send_message(prompt)
    response = api_response.text
    return response     


def query_gemini(model: str, messages: list[dict], temperature: float, 
                 func_name_for_error: str, retry_on_recitation: bool = True, retry_temp: float = 0.1) -> str:
    try:
        model_instance = genai.GenerativeModel(
            model_name=model,
            generation_config={
                'temperature': temperature,
            },
            safety_settings=GEMINI_SAFETY_SETTINGS,
        )
        history = messages[:-1]
        prompt = messages[-1]
        chat = model_instance.start_chat(history=history)
        api_response = chat.send_message(prompt)
        response = api_response.text
        return response
    except Exception as e:
        if 'RECITATION' in str(e) and retry_on_recitation:
            print(colored(f"WARNING in query_gemini: Encountered 'RECITATION' error for model {model}. Retrying with temperature {retry_temp}...", "red"))
            # Call the function again, but if encounter the same error, don't retry, just raise the error
            response = query_gemini(model, messages, temperature=retry_temp, 
                                    func_name_for_error=func_name_for_error, retry_on_recitation=False)
            print(colored(f'SUCCESS: Successfully retried with temperature {retry_temp}.', "green"))
            return response
        elif "quota" in str(e).lower() or '429' in str(e):
            raise CustomRateLimitError(model, func_name_for_error)
        raise
