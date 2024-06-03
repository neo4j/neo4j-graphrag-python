from typing import Optional

from openai import OpenAI

from .base import LLMInterface


class OpenAILLM(LLMInterface):
    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict] = None,
    ):
        """

        Args:
            model_name (str):
            model_params (str): Parameters like temperature and such  that will be
             passed to the model

        """
        super().__init__(model_name, model_params)
        self.client = OpenAI()  # reading the API-KEY from env var OPENAI_API_KEY

    def get_messages(
        self,
        input: str,
    ) -> list:
        return [
            {"role": "system", "content": input},
        ]

    def invoke(self, input: str) -> str:
        response = self.client.chat.completions.create(
            messages=self.get_messages(input),
            model=self.model_name,
            **self.model_params,
        )
        # TODO: deal with errors
        return response.choices[0].message.content
