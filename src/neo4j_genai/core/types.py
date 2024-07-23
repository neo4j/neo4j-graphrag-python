from pydantic import BaseModel, ConfigDict

from neo4j_genai.core.component import Component


class ComponentDef(BaseModel):
    name: str
    component: Component

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ConnectionDef(BaseModel):
    start: str
    end: str
    input_defs: dict[str, str]


class PipelineDef(BaseModel):
    components: list[ComponentDef]
    connections: list[ConnectionDef]
