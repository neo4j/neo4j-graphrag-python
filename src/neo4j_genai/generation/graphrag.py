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
from typing import Optional, Any

from pydantic import ValidationError

from neo4j_genai.exceptions import (
    SearchValidationError,
    RagInitializationError,
)
from neo4j_genai.retrievers.base import Retriever
from neo4j_genai.generation.types import RagResultModel, RagInitModel, RagSearchModel
from neo4j_genai.llm import LLMInterface
from neo4j_genai.generation.prompts import RagTemplate
from neo4j_genai.types import RetrieverResult

logger = logging.getLogger(__name__)


class GraphRAG:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMInterface,
        prompt_template: RagTemplate = RagTemplate(),
    ):
        try:
            validated_data = RagInitModel(
                retriever=retriever,
                llm=llm,
                prompt_template=prompt_template,
            )
        except ValidationError as e:
            raise RagInitializationError(e.errors())
        self.retriever = validated_data.retriever
        self.llm = validated_data.llm
        self.prompt_template = validated_data.prompt_template

    def search(
        self,
        query: str,
        examples: str = "",
        retriever_config: Optional[dict[str, Any]] = None,
        return_context: bool = False,
    ) -> RagResultModel:
        """This method performs a full RAG search:
        1. Retrieval: context retrieval
        2. Augmentation: prompt formatting
        3. Generation: answer generation with LLM

        Args:
            query (str): The user question
            examples: Examples added to the LLM prompt.
            retriever_config (Optional[dict]): Parameters passed to the retriever
                search method; e.g.: top_k
            return_context (bool): Whether to return the retriever result (default: False)

        Returns:
            RagResultModel: The LLM-generated answer

        """
        try:
            validated_data = RagSearchModel(
                query=query,
                examples=examples,
                retriever_config=retriever_config or {},
                return_context=return_context,
            )
        except ValidationError as e:
            raise SearchValidationError(e.errors())
        query = validated_data.query
        retriever_result: RetrieverResult = self.retriever.search(
            query_text=query, **validated_data.retriever_config
        )
        context = "\n".join(item.content for item in retriever_result.items)
        prompt = self.prompt_template.format(
            query=query, context=context, examples=validated_data.examples
        )
        logger.debug(f"RAG: retriever_result={retriever_result}")
        logger.debug(f"RAG: prompt={prompt}")
        answer = self.llm.invoke(prompt)
        result: dict[str, Any] = {"answer": answer.content}
        if return_context:
            result["retriever_result"] = retriever_result
        return RagResultModel(**result)
