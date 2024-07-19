from __future__ import annotations

import asyncio
from typing import Any

from neo4j_genai.core.pipeline import Component


class DocumentChunker(Component):
    async def process(self, text: str) -> dict[str, Any]:
        return {"chunks": [t.strip() for t in text.split(".") if t.strip()]}


class SchemaBuilder(Component):
    async def process(self, schema: dict[str, Any]) -> dict[str, Any]:
        return {"schema": schema}


class ERExtractor(Component):
    async def _process_chunk(self, chunk: str, schema: str) -> dict[str, Any]:
        return {
            "data": {
                "entities": [{"label": "Person", "properties": {"name": "John Doe"}}],
                "relations": [],
            }
        }

    async def process(self, chunks: list[str], schema: str) -> dict[str, Any]:
        tasks = [self._process_chunk(chunk, schema) for chunk in chunks]
        result = await asyncio.gather(*tasks)
        merged_result: dict[str, Any] = {"data": {"entities": [], "relations": []}}
        for res in result:
            merged_result["data"]["entities"] += res["data"]["entities"]
            merged_result["data"]["relations"] += res["data"]["relations"]
        return merged_result


class Writer(Component):
    async def process(
        self, entities: dict[str, Any], relations: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "status": "OK",
            "entities": entities,
            "relations": relations,
        }


if __name__ == "__main__":
    from neo4j_genai.core.pipeline import Pipeline

    pipe = Pipeline()
    pipe.add_component("chunker", DocumentChunker())
    pipe.add_component("schema", SchemaBuilder())
    pipe.add_component("extractor", ERExtractor())
    pipe.add_component("writer", Writer())
    pipe.connect("chunker", "extractor", input_defs={"chunks": "chunker.chunks"})
    pipe.connect("schema", "extractor", input_defs={"schema": "schema.schema"})
    pipe.connect(
        "extractor",
        "writer",
        input_defs={
            "entities": "extractor.data.entities",
            "relations": "extractor.data.relations",
        },
    )

    pipe_inputs = {
        "chunker": {
            "text": "Graphs are everywhere. "
            "GraphRAG is the future of Artificial Intelligence. "
            "Robots are already running the world."
        },
        "schema": {"schema": "Person OWNS House"},
    }
    # print(pipe.run_all(pipe_inputs))
    print(asyncio.run(pipe.run(pipe_inputs)))
