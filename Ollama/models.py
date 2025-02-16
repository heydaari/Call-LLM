from ollama import chat
from ollama import ChatResponse


class OllamaModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.chat_history = []

    def __call__(self, prompt, stream=False, with_history=True):

        if with_history:
            messages = self.chat_history + [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]

        if stream:
            model_stream = chat(model=self.model_name, messages=messages, stream=True)
            return model_stream

        response_text : ChatResponse = chat(model=self.model_name, messages=messages)
        response_text = response_text['message']['content']

        self.chat_history.append({"role": "user", "content": prompt})
        self.chat_history.append({"role": "assistant", "content": response_text})

        return response_text

    def reset_history(self):
        self.chat_history = []