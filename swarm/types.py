from pydantic import BaseModel
from typing import List, Callable, Union, Optional, Dict

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from swarm.memory import (
    ShortTermMemory,
    LongTermMemory,
    EntityMemory,
    ContextualMemory,
)

AgentFunction = Callable[[], Union[str, "Agent", dict]]


class Agent(BaseModel):
    name: str = "Agent"
    model: str = ""
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    functions: List[AgentFunction] = []
    tool_choice: str = "auto"
    parallel_tool_calls: bool = True

    # Memory flags
    memory: bool = False
    use_short_term_memory: bool = False
    use_long_term_memory: bool = False
    use_entity_memory: bool = False

    # Memory instances
    short_term_memory: Optional[ShortTermMemory] = None
    long_term_memory: Optional[LongTermMemory] = None
    entity_memory: Optional[EntityMemory] = None
    contextual_memory: Optional[ContextualMemory] = None

    def initialize_memories(self):
        """Initialize the requested memory types"""
        if self.memory:
            if self.use_long_term_memory and not self.long_term_memory:
                self.long_term_memory = LongTermMemory(
                    agent_name=self.name,
                )

            if self.use_short_term_memory and not self.short_term_memory:
                self.short_term_memory = ShortTermMemory(
                    agent_name=self.name,
                )

            if self.use_entity_memory and not self.entity_memory:
                self.entity_memory = EntityMemory(agent_name=self.name)

            self.contextual_memory = ContextualMemory(
                long_term=self.long_term_memory,
                short_term=self.short_term_memory,
                entity=self.entity_memory,
            )

    def build_context_from_query(self, query: str) -> Dict:
        """Retrieve context-relevant memories"""
        if not self.memory or not self.contextual_memory:
            return {"context": []}

        return self.contextual_memory.synthesize(query)


class Response(BaseModel):
    messages: List = []
    agent: Optional[Agent] = None
    context_variables: dict = {}


class Result(BaseModel):
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    value: str = ""
    agent: Optional[Agent] = None
    context_variables: dict = {}
