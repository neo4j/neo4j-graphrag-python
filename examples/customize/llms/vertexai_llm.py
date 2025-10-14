from neo4j_graphrag.llm import LLMResponse, VertexAILLM
from vertexai.generative_models import GenerationConfig

generation_config = GenerationConfig(temperature=1.0)
llm = VertexAILLM(
    model_name="gemini-2.0-flash-001",
    generation_config=generation_config,
    # add here any argument that will be passed to the
    # vertexai.generative_models.GenerativeModel client
)
res: LLMResponse = llm.invoke(
    "say something",
    system_instruction="You are living in 3000 where AI rules the world",
)
print(res.content)
