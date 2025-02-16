from huggingface_hub import InferenceClient

class HuggingFaceInference:
    def __init__(self,
                 model_name: str,
                 key: str):

        """
        Initializes the HuggingFaceChat client.

        Args:
            model_name: The name of the Hugging Face model to use.
            key: Your Hugging Face API key.
        """

        self.client = InferenceClient(model=model_name, api_key=key)
        self.model_name = model_name
        self.api_key = key
        self.chat_history = []

    def __call__(self,
                 prompt: str,
                 stream: bool = True,
                 max_tokens: int = 1000,
                 temperature: float = 0.001,
                 with_history: bool = True):

        """
        Sends a prompt to the Hugging Face model and returns the response.

        Args:
            prompt: The user's input prompt.
            stream: Whether to stream the response (default: True).
            max_tokens: The maximum number of tokens in the response (default: 1000).
            temperature: The temperature for sampling (default: 0.001).
            with_history: Whether to include chat history in the prompt (default: True).

        Returns:
            If stream is True, yields response chunks. Otherwise, returns the full response text.
        """

        if with_history:
            messages = self.chat_history + [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]

        try:
            output = self.client.chat.completions.create(
                messages=messages,
                stream=stream,
                max_tokens=max_tokens,
                temperature=temperature
            )

            if stream:
                full_response = ""

                for chunk in output:
                    content = chunk.choices[0].delta.content or ""
                    full_response += content
                    yield content

                self.chat_history.append({"role": "user", "content": prompt})
                self.chat_history.append({"role": "assistant", "content": full_response})

            else:
                response_text = "".join([chunk.choices[0].delta.content or "" for chunk in output])

                self.chat_history.append({"role": "user", "content": prompt})
                self.chat_history.append({"role": "assistant", "content": response_text})

                return response_text

        except Exception as e:
            print(f"Error during inference: {e}")
            yield f"Error: {e}"  

    def reset_chat(self):
        """Resets the chat history."""
        self.chat_history = []