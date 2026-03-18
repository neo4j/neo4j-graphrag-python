# Structured Output in SimpleKGPipeline

The neo4j-graphrag-python library (v1.13.0+) replaced prompt-based JSON extraction with LLM-enforced structured output for knowledge graph construction. The extraction pipeline no longer asks the LLM to produce JSON and hopes the format is correct. Instead, it passes Pydantic model schemas directly to the LLM API, which constrains generation to valid output at decode time.

## The Problem with Prompt-Based Extraction

Prior to v1.13.0, entity and relation extraction followed a two-step pattern: prompt the LLM with JSON formatting instructions, then repair whatever came back. The `fix_invalid_json` function, `balance_curly_braces`, and the `json_repair` library all exist because LLMs routinely produce malformed JSON when given format instructions in a prompt. Missing closing braces, trailing commas, unescaped strings. Each chunk extraction was a coin flip between clean output and a cascade through error-handling fallbacks.

Schema extraction from text had the same fragility. The LLM would receive instructions to output node types, relationship types, and patterns as JSON. The response required cleaning, normalization, and multiple validation passes before it could be trusted.

## LLMInterfaceV2

The foundation is a new LLM interface that accepts a `response_format` parameter alongside the message list:

```python
class LLMInterfaceV2(ABC):
    @abstractmethod
    async def ainvoke(
        self,
        input: List[LLMMessage],
        response_format: Optional[Union[Type[BaseModel], dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
```

When `response_format` receives a Pydantic model class, the LLM provider converts it to a JSON schema and passes it to the API's native structured output mechanism. OpenAI wraps the schema in a `json_schema` response format with `strict: True`. VertexAI sets `response_mime_type` to `application/json` and passes the schema to `GenerationConfig`.

The old `LLMInterface` (V1) accepted a plain string input and returned unstructured text. It remains available but logs a deprecation warning on instantiation.

Each LLM declares its capability through a class attribute:

```python
supports_structured_output: bool = False  # default on LLMInterface
```

OpenAI and VertexAI set this to `True`. Anthropic, Cohere, Mistral, Ollama, and Bedrock leave it `False`.

## Two Pydantic Models Drive the Pipeline

### Neo4jGraph (entity/relation extraction)

Defined in `experimental/components/types.py`, this model captures the output of each chunk's extraction:

```python
class Neo4jNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    label: str
    properties: dict[str, PropertyValue] = Field(default_factory=dict)
    embedding_properties: dict[str, list[float]] = Field(default_factory=dict)

class Neo4jRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_node_id: str
    end_node_id: str
    type: str
    properties: dict[str, PropertyValue] = Field(default_factory=dict)
    embedding_properties: dict[str, list[float]] = Field(default_factory=dict)

class Neo4jGraph(DataModel):
    model_config = ConfigDict(extra="forbid")
    nodes: list[Neo4jNode] = Field(default_factory=list)
    relationships: list[Neo4jRelationship] = Field(default_factory=list)
```

`Neo4jGraph` overrides `model_json_schema` to force `nodes` and `relationships` into the `required` array. Without this override, Pydantic marks fields with defaults as optional, and the LLM may omit them entirely.

### GraphSchema (schema-from-text extraction)

Defined in `experimental/components/schema.py`, this model captures the schema that guides extraction. Its `model_json_schema` override is more involved because it must satisfy constraints from two different providers:

- OpenAI requires `additionalProperties: false` on every object and all properties listed in `required`.
- VertexAI rejects the `const` keyword in JSON schemas; it must be converted to a single-value `enum`.

A recursive `make_strict` function walks the schema tree and applies both transformations, making the same Pydantic model work across providers without per-provider schema variants.

## How the Extractor Switches Between Modes

`LLMEntityRelationExtractor` accepts a `use_structured_output` flag. The constructor validates that the LLM actually supports it:

```python
if use_structured_output and not llm.supports_structured_output:
    raise ValueError(
        f"Structured output is not supported by {type(llm).__name__}."
    )
```

At extraction time, the two paths diverge completely. V2 passes the Pydantic model as a response format and validates the response with `model_validate_json`. V1 calls `ainvoke` with a plain string, repairs the JSON, parses it, and then validates:

```python
# V2: structured output
messages = [LLMMessage(role="user", content=prompt)]
llm_result = await self.llm.ainvoke(messages, response_format=Neo4jGraph)
chunk_graph = Neo4jGraph.model_validate_json(llm_result.content)

# V1: prompt-based
llm_result = await self.llm.ainvoke(prompt)
llm_generated_json = fix_invalid_json(llm_result.content)
result = json.loads(llm_generated_json)
chunk_graph = Neo4jGraph.model_validate(result)
```

The same dual-path pattern exists in `SchemaFromTextExtractor` for schema extraction from text.

## SimpleKGPipeline Auto-Detection

`SimpleKGPipelineConfig` queries the LLM's capability flag and enables structured output automatically. No user configuration required:

```python
def _get_schema(self) -> BaseSchemaBuilder:
    if not self.has_user_provided_schema():
        llm = self.get_default_llm()
        return SchemaFromTextExtractor(
            llm=llm,
            use_structured_output=llm.supports_structured_output,
        )
    return SchemaBuilder()

def _get_extractor(self) -> EntityRelationExtractor:
    llm = self.get_default_llm()
    return LLMEntityRelationExtractor(
        llm=llm,
        use_structured_output=llm.supports_structured_output,
    )
```

Pass an OpenAI or VertexAI LLM to `SimpleKGPipeline` and both schema extraction and entity/relation extraction use structured output. Pass any other provider and the pipeline falls back to V1 prompt-based extraction transparently.

## Provider Support

| Provider | Structured Output | Interface |
|----------|:-:|:-:|
| OpenAI | Yes | LLMInterfaceV2 |
| VertexAI | Yes | LLMInterfaceV2 |
| Anthropic | No | LLMInterface (V1) |
| Cohere | No | LLMInterface (V1) |
| Mistral | No | LLMInterface (V1) |
| Ollama | No | LLMInterface (V1) |
| Bedrock | No | LLMInterface (V1) |

## Relevant Commits

- `57f80e6` - LLMInterfaceV2 and LLMEntityRelationExtractor structured output support
- `02d4d82` - SchemaFromTextExtractor structured output support
- `3c3a231` - SimpleKGPipeline auto-detection wiring
