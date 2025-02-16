import google.generativeai as genai

class Gemini:
    def __init__(self,
                 model_name : str,
                 key : str,
                 tools : list = None):

        genai.configure(api_key=key)

        self.model = genai.GenerativeModel(model_name= model_name, tools= tools)
        self.model_name = model_name
        self.key = key
        self.chat = None

    def __call__(self,
                 prompt: str,
                 with_history: bool = True,
                 generation_config: dict = None):

        """
        Calls the Gemini model to generate content.

        Args:
            prompt: The text prompt to send to the model.
            with_history: A boolean indicating whether to use the previous session history.
                         Defaults to True.
            generation_config: A dictionary containing generation parameters like temperature, top_p, and max_output_tokens.
                               Defaults to None (using model defaults).

        Returns:
            The generated content from the model.
        """

        if generation_config is None:
            generation_config = {}

        if with_history:
            if self.chat is None:
                self.chat = self.model.start_chat()

            response = self.chat.send_message(prompt)

        else:
            response = self.model.generate_content(prompt, generation_config=generation_config)

        return response.text

    def reset_chat(self):
        """
        Resets the chat session, clearing the history.
        """
        self.chat = None