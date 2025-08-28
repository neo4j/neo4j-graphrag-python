from neo4j_graphrag.llm import LLMResponse, VertexAILLM
from vertexai.generative_models import GenerationConfig

from neo4j_graphrag.types import LLMMessage

messages: list[LLMMessage] = [
    {
        "role": "system",
        "content": "You are a seasoned actor and expert performer, renowned for your one-man shows and comedic talent.",
    },
    {
        "role": "user",
        "content": "say something",
    },
]


generation_config = GenerationConfig(temperature=1.0)
llm = VertexAILLM(
    model_name="gemini-2.0-flash-001",
    generation_config=generation_config,
    # add here any argument that will be passed to the
    # vertexai.generative_models.GenerativeModel client
)
res: LLMResponse = llm.invoke(
    input=messages,
)
print(res.content)
