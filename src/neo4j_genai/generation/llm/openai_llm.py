#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
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
