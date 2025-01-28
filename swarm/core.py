from abc import ABC, abstractmethod
from typing import List, Any

from .types import (
    Agent,
    Response,
    Result
)

class Swarm(ABC):
    @abstractmethod
    def get_chat_completion(
        self,
        agent: Agent,
        history: List,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> Any:
        pass

    @abstractmethod 
    def handle_function_result(self, result, debug) -> Result:
        pass

    @abstractmethod
    def handle_tool_calls(
        self,
        tool_calls: List,
        functions: List,
        context_variables: dict,
        debug: bool,
    ) -> Response:
        pass

    @abstractmethod
    def run_and_stream(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Any:
        pass

    @abstractmethod
    def run(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        pass