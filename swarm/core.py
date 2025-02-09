import copy
import os
import json
from datetime import datetime
from collections import defaultdict
from typing import List
from litellm import completion
import litellm
from .tasks import Task
from .util import function_to_json, debug_print, merge_chunk
from .types import (
    Agent,
    AgentFunction,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Function,
    Response,
    Result,
)
from .memory import (
    LongTermMemory,
    LongTermMemoryItem,
    ShortTermMemory,
    ShortTermMemoryItem,
    EntityMemory,
    EntityMemoryItem,
    ContextualMemory
)
from swarm.utilities import TaskEvaluation

__CTX_VARS_NAME__ = "context_variables"
litellm.drop_params = True


class Swarm():
    def __init__(self, memory: bool = False):
        self.memory = memory
        self.create_memory()

    def create_memory(self):
        if self.memory:
            # NEED TO UPDATE
            embedder_config = {
                "provider": "google",
                "config": {
                    "model": "models/text-embedding-004",
                    "api_key": os.getenv("GEMINI_API_KEY")
                }
            }
            self._long_term_memory = LongTermMemory()
            self._short_term_memory = ShortTermMemory(
                embedder_config=embedder_config)
            self._entity_memory = EntityMemory(embedder_config=embedder_config)

    def _build_context_from_message(self, message):
        if not self.memory:
            return message

        contextual_memory = ContextualMemory(
            ltm=self._long_term_memory,
            stm=self._short_term_memory,
            em=self._entity_memory
        )
        # TODO: update context here?
        context = ""

        return contextual_memory.build_context_for_task(self.task, context)

    def get_chat_completion(
        self,
        agent: Agent,
        history: List,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> ChatCompletionMessage:
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        messages = [{"role": "system", "content": instructions}] + history
        debug_print(debug, "Getting chat completion for...:", messages)

        tools = [function_to_json(f) for f in agent.functions]
        # hide context_variables from model
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)

        if not agent.model and not model_override:
            raise ValueError(
                "Please provide either the agent model name or model_override.")

        create_params = {
            "model": model_override or agent.model,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
            "stream": stream,
        }
        model_config = agent.model_config
        create_params.update(model_config)

        try:
            response = completion(**create_params)
            return response
        except:
            error_message = "Please verify that an API key is provided, or errors may occur due to model limitations."
            raise ValueError(error_message)

    def handle_function_result(self, result, debug) -> Result:
        match result:
            case Result() as result:
                return result

            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": agent.name}),
                    agent=agent,
                )
            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                    debug_print(debug, error_message)
                    raise TypeError(error_message)

    def handle_tool_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        functions: List[AgentFunction],
        context_variables: dict,
        debug: bool,
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(
            messages=[], agent=None, context_variables={})

        for tool_call in tool_calls:
            name = tool_call.function.name

            # handle missing tool case, skip to next tool
            if name not in function_map:
                debug_print(debug, f"Tool {name} not found in function map.")
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "tool_name": name,
                        "content": f"Error: Tool {name} not found.",
                    }
                )
                continue

            args = json.loads(tool_call.function.arguments)
            debug_print(
                debug, f"Processing tool call: {name} with arguments {args}")

            func = function_map[name]
            # pass context_variables to agent functions
            if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                args[__CTX_VARS_NAME__] = context_variables

            valid_params = function_map[name].__code__.co_varnames[:
                                                                   function_map[name].__code__.co_argcount]
            filtered_args = {k: v for k,
                             v in args.items() if k in valid_params}

            raw_result = function_map[name](**filtered_args)

            result: Result = self.handle_function_result(raw_result, debug)

            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": name,
                    "content": result.value,
                }
            )

            partial_response.context_variables.update(result.context_variables)
            if result.agent:
                partial_response.agent = result.agent

        return partial_response

    def run_and_stream(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ):
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns:

            message = {
                "content": "",
                "sender": agent.name,
                "role": "assistant",
                "function_call": None,
                "tool_calls": defaultdict(
                    lambda: {
                        "function": {"arguments": "", "name": ""},
                        "id": "",
                        "type": "",
                    }
                ),
            }

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=True,
                debug=debug,
            )

            yield {"delim": "start"}
            for chunk in completion:
                delta = json.loads(chunk.choices[0].delta.json())
                if delta["role"] == "assistant":
                    delta["sender"] = active_agent.name
                yield delta
                delta.pop("role", None)
                delta.pop("sender", None)
                merge_chunk(message, delta)
            yield {"delim": "end"}

            message["tool_calls"] = list(
                message.get("tool_calls", {}).values())
            if not message["tool_calls"]:
                message["tool_calls"] = None
            debug_print(debug, "Received completion:", message)
            history.append(message)

            if not message["tool_calls"] or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # convert tool_calls to objects
            tool_calls = []
            for tool_call in message["tool_calls"]:
                function = Function(
                    arguments=tool_call["function"]["arguments"],
                    name=tool_call["function"]["name"],
                )
                tool_call_object = ChatCompletionMessageToolCall(
                    id=tool_call["id"], function=function, type=tool_call["type"]
                )
                tool_calls.append(tool_call_object)

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        yield {
            "response": Response(
                messages=history[init_len:],
                agent=active_agent,
                context_variables=context_variables,
            )
        }

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
        self.task = Task(description=messages[-1]["content"])
        context_memory = self._build_context_from_message(messages)
        if context_memory:
            messages[-1]["content"] += (
                "Here is the context for the task"
                f"{context_memory}"
            )

        if stream:
            return self.run_and_stream(
                agent=agent,
                messages=messages,
                context_variables=context_variables,
                model_override=model_override,
                debug=debug,
                max_turns=max_turns,
                execute_tools=execute_tools,
            )
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns and active_agent:

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
                debug=debug,
            )

            message = completion.choices[0].message
            debug_print(debug, "Received completion:", message)
            message.sender = active_agent.name

            history.append(
                json.loads(message.model_dump_json())
            )

            if not message.tool_calls or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                message.tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent
        response = Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )
        if self.memory:
            self._udpate_long_term_memory(response)
            # NOTE: consider adding short term memory
        return response

    def evaluate_task(self, task: Task, response: Response):
        messages = response.messages
        evaluation_query = (
            f"Assess the quality of the task completed based on the description, expected output, and actual message.\n\n"
            f"Task Description:\n{task.description}\n\n"
            f"Expected Output:\n{task.expected_output}\n\n" if hasattr(
                task, "expected_output") else ""
            f"Actual Output:\n{messages}\n\n"
            "Please provide:\n"
            "- Bullet points suggestions to improve future similar tasks\n"
            "- A score from 0 to 10 evaluating on completion, quality, and overall performance"
            "- Entities extracted from the task output, if any, their type, description, and relationships"
        )

        instructions = (
            "Convert all responses into valid JSON output follow by the following schema."
            f"{TaskEvaluation.schema_json(indent=2)}"
        )

        content = (
            f"{evaluation_query}\n\n"
            f"{instructions}"
        )
        response = completion(
            model=response.agent.model,
            messages=[{"content": content, "role": "user"}],
            response_format={
                "type": "json_object",
            }
        )
        json_response = response.choices[0].message.content
        evaluation = TaskEvaluation.parse_raw(json_response)

        return evaluation

    def _udpate_long_term_memory(self, output) -> None:
        """Create and save long-term and entity memory items based on evaluation."""
        if (
            self.memory
            and self._long_term_memory
            and self._entity_memory
        ):
            try:
                evaluation = self.evaluate_task(self.task, output)
                long_term_memory = LongTermMemoryItem(
                    task=self.task.description,
                    agent=output.agent.name,
                    quality=evaluation.quality,
                    datetime=str(datetime.now()),
                    expected_output="" if not hasattr(
                        self.task, "expected_output") else self.task.expected_output,
                    metadata={
                        "suggestions": evaluation.suggestions,
                        "quality": evaluation.quality,
                    },
                )
                self._long_term_memory.save(long_term_memory)
                for entity in evaluation.entities:
                    entity_memory = EntityMemoryItem(
                        name=entity.name,
                        type=entity.type,
                        description=entity.description,
                        relationships="\n".join(
                            [f"- {r}" for r in entity.relationships]
                        ),
                    )
                    self._entity_memory.save(entity_memory)
            except AttributeError as e:
                print(f"Missing attributes for long term memory: {e}")
            except Exception as e:
                print(f"Failed to add to long term memory: {e}")

    def _create_short_term_memory(self, output) -> None:
        """Create and save a short-term memory item if conditions are met."""
        if (self.memory):
            try:
                # TODO: update mechanism to extract value from output
                value = output.messages[-1]["content"]
                self._short_term_memory.save(
                    value=value,
                    metadata={
                        "observation": self.task.description,
                    },
                    agent=output.agent.name,
                )
            except Exception as e:
                print(f"Failed to add to short term memory: {e}")
