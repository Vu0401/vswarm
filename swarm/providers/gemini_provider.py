import os
from typing import List
from collections import defaultdict
import copy
import json

import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from google.generativeai.types import content_types

from ..core import Swarm
from ..util import debug_print
from ..types import (
    Agent,
    AgentFunction,
    Response,
    Result,
)
from dotenv import load_dotenv
load_dotenv()


__CTX_VARS_NAME__ = "context_variables"


class GeminiSwarm(Swarm):
    def __init__(self, client=None, GEMINI_API_KEY=None):
        # gemini api key
        genai.configure(api_key=GEMINI_API_KEY) 
               
        # Gemini default client 
        self.client = genai.GenerativeModel() if not client else client
        
    def get_chat_completion( 
        self,
        agent: Agent,
        user_input: str,
        history: list,
        generation_config: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> list:
        
        if stream or debug:
            pass
        
        instructions = agent.instructions
        
        if not generation_config:
            generation_config = {
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 50,
            "response_mime_type": "text/plain",
            }
        
        tools = [f for f in agent.functions]

        setattr(self.client, '_model_name', f"models/{model_override or agent.model or 'gemini-1.5-flash'}")  
        setattr(self.client, '_system_instruction', content_types.to_content(instructions)) 
        setattr(self.client, '_tools', tools) 
        setattr(self.client, '_generation_config', generation_config)  

        chat_session = self.client.start_chat(history=history) 
        response = chat_session.send_message(user_input)
        parts = response.candidates[0].content.parts

        return parts

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
        completion: list,
        functions: List[AgentFunction],
        debug: bool,
    ) -> Response:
        
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(
            messages=[], agent=None, context_variables={})

        for tool in completion:
            function_call = tool.function_call
            name = function_call.name
            
            # handle missing tool case, skip to next tool
            if name not in function_map:
                partial_response.messages.append(
                    {
                        "role": "user", 
                        "parts": str(   
                                    {
                                    "role": "tool", 
                                    "tool_name":name, 
                                    "content":f"Error: Tool {name} not found."})
                    }
                )
                continue
            
            args = {key: value for key, value in function_call.args.items()}
            func = function_map[name]
            raw_result = func(**args)
            result: Result = self.handle_function_result(raw_result, debug)

            partial_response.messages.append(
                {
                    "role": "user", 
                    "parts":str(    
                                {
                                "role": "tool", 
                                "tool_name":name, 
                                "args":args, 
                                "content":result.value}
                                )
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
        pass

    def run(
        self,
        agent: Agent,
        messages: List = [],
        generation_config: dict = None,
        model_override: str = None,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        
        if stream:
            pass
            
        active_agent = agent
        context_variables = {}
        history = copy.deepcopy(messages) # messages = [{"role": "user", "parts":"hello"}...]
        user_input = "Answer the question: " + history[-1]["parts"]
        init_len = len(messages)
        
        while len(history) - init_len < max_turns and active_agent:
            # get completion (Gemini parts) with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                user_input=user_input,
                history=history,
                generation_config=generation_config,
                model_override=model_override,
                stream=stream,
                debug=debug,
            )
            if completion[0].text:
                history.append(
                                {
                                "role": "model", 
                                "parts":completion[0].text
                                }
                            ) 
                
            if not completion[0].function_call or not execute_tools:
                break
            
            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(completion, active_agent.functions, debug)
            
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )
