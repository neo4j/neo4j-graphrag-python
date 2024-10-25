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

from typing import Any

from neo4j_graphrag.embeddings.base import Embedder


class SentenceTransformerEmbeddings(Embedder):
    def __init__(
        self, model: str = "all-MiniLM-L6-v2", *args: Any, **kwargs: Any
    ) -> None:
        try:
            import numpy as np
            import sentence_transformers
            import torch
        except ImportError:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            )
        self.torch = torch
        self.np = np
        self.model = sentence_transformers.SentenceTransformer(model, *args, **kwargs)

    def embed_query(self, text: str) -> Any:
        result = self.model.encode([text])
        if isinstance(result, self.torch.Tensor) or isinstance(result, self.np.ndarray):
            return result.flatten().tolist()
        elif isinstance(result, list) and all(
            isinstance(x, self.torch.Tensor) for x in result
        ):
            return [item for tensor in result for item in tensor.flatten().tolist()]
        else:
            raise ValueError("Unexpected return type from model encoding")
