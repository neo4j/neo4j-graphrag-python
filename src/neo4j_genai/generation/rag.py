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
import logging
from typing import Optional
from ..retrievers.base import Retriever
from .llm import LLMInterface
from .prompts import RagTemplate


logger = logging.getLogger(__name__)


class RAG:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMInterface,
        prompt_template: RagTemplate = RagTemplate(),
    ):
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template

    def search(self, query: str, retriever_config: Optional[dict] = None) -> str:
        """
        Args:
            query (str): The user question
        """
        retriever_config = retriever_config or {}
        retriever_result = self.retriever.search(query_text=query, **retriever_config)
        context = "\n".join(item.content for item in retriever_result.items)
        prompt = self.prompt_template.format(query=query, context=context)
        logger.debug(f"RAG: context={context}, prompt={prompt}")
        return self.llm.invoke(prompt)
