from swarm.utilities.printer import Printer
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

printer = Printer()

class Swarm():
    def _build_agent_context(self, agent: Agent, query: str) -> str:
        """Build context from agent's active memories"""
        if not agent.memory:
            return ""

        context = agent.build_context_from_query(query)
        if not context or not context.get("context"):
            return ""

        context_text = "\nRelevant context from memory:\n"
        for item in context.get("context", []):
            context_text += f"- {item}\n"

        return context_text

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
        """
        Execute the agent with the given messages and handle memory operations.
        """
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

        # Initialize agent memories if needed
        if agent.memory and any([agent.use_entity_memory, agent.use_long_term_memory, agent.use_short_term_memory]):
            agent.initialize_memories()

        active_agent = agent
        previous_agent = None
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        # Get the main task description from the last user message
        task_description = messages[-1].get("content", "") if messages else ""

        while len(history) - init_len < max_turns and active_agent:
            # Track agent changes
            if previous_agent and previous_agent != active_agent:
                # Update memories for the previous agent before switching
                partial_response = Response(
                    messages=history[init_len:],
                    agent=previous_agent,
                    context_variables=context_variables,
                )
                self._update_memories(
                    previous_agent, task_description, partial_response)

                # Initialize memories for new agent if needed
                if active_agent.memory and any([
                    active_agent.use_entity_memory,
                    active_agent.use_long_term_memory,
                    active_agent.use_short_term_memory
                ]):
                    active_agent.initialize_memories()

            # Build context from active agent's memories
            memory_context = self._build_agent_context(
                active_agent, task_description)
            if memory_context:
                last_msg = history[-1].copy()
                last_msg["content"] = f"{last_msg.get('content', '')}\n{memory_context}"
                history[-1] = last_msg

            if memory_context:
                printer.print(memory_context, 'green') 
            else:
                printer.print("No relevant context found in memory.", 'red')    
                
            # Get completion with current history, agent
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

            history.append(json.loads(message.model_dump_json()))

            if not message.tool_calls or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # Store current agent before potential change
            previous_agent = active_agent

            # Handle function calls
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

        # Update memories for the final active agent
        self._update_memories(active_agent, task_description, response)

        return response

    def _update_memories(self, agent: Agent, task_description: str, response: Response) -> None:
        """Update agent's memories based on the interaction"""
        if not agent.memory:
            return

        try:
            # Evaluate task if long-term or entity memory is used
            if agent.use_long_term_memory or agent.use_entity_memory:
                evaluation = self._evaluate_response(
                    task_description, response)

                # Update long-term memory
                if agent.use_long_term_memory:
                    self._update_long_term_memory(
                        agent, task_description, evaluation)

                # Update entity memory
                if agent.use_entity_memory:
                    self._update_entity_memory(agent, evaluation)

            # Update short-term memory
            if agent.use_short_term_memory:
                self._update_short_term_memory(
                    agent, task_description, response)

        except Exception as e:
            print(f"Error updating memories: {str(e)}")

    def _evaluate_response(self, task_description: str, response: Response) -> TaskEvaluation:
        """Evaluate the task completion and return structured feedback"""
        evaluation_query = (
            f"Assess the quality of the task completed based on the description and actual messages.\n\n"
            f"Task Description:\n{task_description}\n\n"
            f"Actual Output:\n{response.messages}\n\n"
            "Please provide:\n"
            "- Bullet points suggestions to improve future similar tasks\n"
            "- A score from 0 to 10 evaluating completion, quality, and overall performance\n"
            "- Entities extracted from the task output, if any, their type, description, and relationships"
        )

        instructions = (
            "Convert all responses into valid JSON output following this schema:\n"
            f"{TaskEvaluation.schema_json(indent=2)}"
        )

        content = f"{evaluation_query}\n\n{instructions}"

        result = completion(
            model=response.agent.model,
            messages=[{"content": content, "role": "user"}],
            response_format={"type": "json_object"}
        )
        
        return TaskEvaluation.parse_raw(result.choices[0].message.content)

    def _update_long_term_memory(self, agent: Agent, task: str, evaluation: TaskEvaluation) -> None:
        """Update long-term memory with task results"""
        memory_item = LongTermMemoryItem(
            task=task,
            agent=agent.name,
            quality=evaluation.quality,
            datetime=str(datetime.now()),
            metadata={
                "suggestions": evaluation.suggestions,
                "quality": evaluation.quality
            }
        )
        printer.print(f"Updating long-term memory\n\n{json.dumps(vars(memory_item), indent=4)}", 'cyan')
        agent.long_term_memory.save(memory_item)

    def _update_entity_memory(self, agent: Agent, evaluation: TaskEvaluation) -> None:
        """Update entity memory with extracted entities"""
        printer.print("Updating entity memory...\n\n", 'cyan')
        for entity in evaluation.entities:
            memory_item = EntityMemoryItem(
                name=entity.name,
                type=entity.type,
                description=entity.description,
                relationships=entity.relationships
            )
            printer.print(f"{json.dumps(vars(memory_item), indent=4)}\n\n", 'cyan')
            agent.entity_memory.save(memory_item)

    def _update_short_term_memory(self, agent: Agent, task: str, response: Response) -> None:
        """Update short-term memory with recent interaction"""
        last_message = response.messages[-1].get(
            "content", "") if response.messages else ""
        memory_item = ShortTermMemoryItem(
            data=last_message,
            metadata={
                "task": task,
                "timestamp": str(datetime.now())
            }
        )
        printer.print(f"Updating short-term memory\n\n{json.dumps(vars(memory_item), indent=4)}", 'cyan')
        agent.short_term_memory.save(memory_item)
