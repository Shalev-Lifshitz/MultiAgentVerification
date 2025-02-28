import datetime


class CustomRateLimitError(Exception):
    def __init__(self, model, function, original_message=None):
        self.model = model
        self.function = function
        self.original_message = original_message
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        super().__init__(f"[{datetime_str} ERROR] Rate limit exceeded for model {model} in {function}")


class CustomJSONDecodeError(Exception):
    def __init__(self, problem_filepath, file_size):
        self.problem_filepath = problem_filepath
        self.file_size = file_size
        super().__init__(f"Encountered JSONDecodeError when loading problem data (file size: {self.file_size}) from {problem_filepath}")
