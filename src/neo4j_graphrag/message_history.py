from abc import ABC, abstractmethod
from typing import List, Optional, Union

import neo4j
from pydantic import PositiveInt

from neo4j_graphrag.llm.types import (
    LLMMessage,
)
from neo4j_graphrag.types import (
    Neo4jDriverModel,
    Neo4jMessageHistoryModel,
)

CREATE_SESSION_NODE_QUERY = "MERGE (s:`{node_label}` {{id:$session_id}})"

CLEAR_SESSION_QUERY = (
    "MATCH (s:`{node_label}`)-[:LAST_MESSAGE]->(last_message) "
    "WHERE s.id = $session_id MATCH p=(last_message)<-[:NEXT]-() "
    "WITH p, length(p) AS length ORDER BY length DESC LIMIT 1 "
    "UNWIND nodes(p) as node DETACH DELETE node;"
)

GET_MESSAGES_QUERY = (
    "MATCH (s:`{node_label}`)-[:LAST_MESSAGE]->(last_message) "
    "WHERE s.id = $session_id MATCH p=(last_message)<-[:NEXT*0.."
    "{window}]-() WITH p, length(p) AS length "
    "ORDER BY length DESC LIMIT 1 UNWIND reverse(nodes(p)) AS node "
    "RETURN {{data:{{content: node.content}}, role:node.role}} AS result"
)

ADD_MESSAGE_QUERY = (
    "MATCH (s:`{node_label}`) WHERE s.id = $session_id "
    "OPTIONAL MATCH (s)-[lm:LAST_MESSAGE]->(last_message) "
    "CREATE (s)-[:LAST_MESSAGE]->(new:Message) "
    "SET new += {{role:$role, content:$content}} "
    "WITH new, lm, last_message WHERE last_message IS NOT NULL "
    "CREATE (last_message)-[:NEXT]->(new) "
    "DELETE lm"
)


class MessageHistory(ABC):
    @property
    @abstractmethod
    def messages(self) -> List[LLMMessage]: ...

    @abstractmethod
    def add_message(self, message: LLMMessage) -> None: ...

    def add_messages(self, messages: List[LLMMessage]) -> None:
        for message in messages:
            self.add_message(message)

    @abstractmethod
    def clear(self) -> None: ...


class InMemoryMessageHistory(MessageHistory):
    def __init__(self, messages: Optional[List[LLMMessage]] = None) -> None:
        self._messages = messages or []

    @property
    def messages(self) -> List[LLMMessage]:
        return self._messages

    def add_message(self, message: LLMMessage) -> None:
        self._messages.append(message)

    def add_messages(self, messages: List[LLMMessage]) -> None:
        self._messages.extend(messages)

    def clear(self) -> None:
        self._messages = []


class Neo4jMessageHistory(MessageHistory):
    def __init__(
        self,
        session_id: Union[str, int],
        driver: neo4j.Driver,
        node_label: str = "Session",
        window: Optional[PositiveInt] = None,
    ) -> None:
        validated_data = Neo4jMessageHistoryModel(
            session_id=session_id,
            driver_model=Neo4jDriverModel(driver=driver),
            node_label=node_label,
            window=window,
        )
        self._driver = validated_data.driver_model.driver
        self._session_id = validated_data.session_id
        self._node_label = validated_data.node_label
        self._window = (
            "" if validated_data.window is None else validated_data.window - 1
        )
        # Create session node
        self._driver.execute_query(
            query_=CREATE_SESSION_NODE_QUERY.format(node_label=self._node_label),
            parameters_={"session_id": self._session_id},
        )

    @property
    def messages(self) -> List[LLMMessage]:
        result = self._driver.execute_query(
            query_=GET_MESSAGES_QUERY.format(
                node_label=self._node_label, window=self._window
            ),
            parameters_={"session_id": self._session_id},
        )
        messages = [
            LLMMessage(
                content=el["result"]["data"]["content"],
                role=el["result"]["role"],
            )
            for el in result.records
        ]
        return messages

    @messages.setter
    def messages(self, messages: List[LLMMessage]) -> None:
        raise NotImplementedError(
            "Direct assignment to 'messages' is not allowed."
            " Use the 'add_messages' instead."
        )

    def add_message(self, message: LLMMessage) -> None:
        self._driver.execute_query(
            query_=ADD_MESSAGE_QUERY.format(node_label=self._node_label),
            parameters_={
                "role": message["role"],
                "content": message["content"],
                "session_id": self._session_id,
            },
        )

    def clear(self) -> None:
        self._driver.execute_query(
            query_=CLEAR_SESSION_QUERY.format(node_label=self._node_label),
            parameters_={"session_id": self._session_id},
        )

    def __del__(self) -> None:
        if self._driver:
            self._driver.close()
