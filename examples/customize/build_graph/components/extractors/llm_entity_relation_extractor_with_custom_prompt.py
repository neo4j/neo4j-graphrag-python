from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
)
from neo4j_graphrag.experimental.components.types import (
    Neo4jGraph,
    TextChunk,
    TextChunks,
)
from neo4j_graphrag.llm import LLMInterface


async def main(llm: LLMInterface) -> Neo4jGraph:
    """

    Args:
        llm (LLMInterface): Any LLM implemented in neo4j_graphrag.llm or from LangChain chat models.
    """
    extractor = LLMEntityRelationExtractor(
        llm=llm,
        # optional: customize the prompt used for entity and relation extraction
        # prompt_template="",
        # optional: disable the creation of the lexical graph (Document and Chunk nodes)
        # create_lexical_graph=False,
        # optional: if an LLM error happens, ignore the chunk and continue process with the next ones
        # default value is OnError.RAISE which will end the process
        # on_error=OnError.IGNORE,
        # optional: tune the max_concurrency parameter to optimize speed
        # max_concurrency=5,
    )
    graph = await extractor.run(chunks=TextChunks(chunks=[TextChunk(text="....")]))
    return graph
