from typing import Callable, Dict, List, Tuple, Dict, Type

from langgraph.graph.graph import CompiledGraph

from langgraph_swarm.util import create_default_llm, default_print_messages

from .router import __CTX_VARS_NAME__
from .types import Agent, Response, HandoffsState

WORKFOLWS = ["DEFAULT", "V03"]

WORKFOLW = WORKFOLWS[0]
if WORKFOLW == "DEFAULT":
    from langgraph_swarm.router import create_swarm_workflow as create_swarm_workflow
else:
    # TODO: we will deprecate this in the future
    from langgraph_swarm.router_with_v03 import create_swarm_workflow as create_swarm_workflow


class Swarm:
    def __init__(self, 
        agent: Agent,
        llm=None,
        state_scheme: Type[HandoffsState] = HandoffsState,
        debug: bool = False,
        print_messages: None | Callable = None,
    ):
        self.llm = llm
        self.print_messages = print_messages # or default_print_messages
        self.debug = debug
        self.state_scheme = state_scheme
        self.start_agent = agent
        self.agent = agent
        self.agent_map = {}
        if agent:
            self.workflow = self._create_workflow()

    def _create_workflow(self) -> CompiledGraph:
        llm = self.llm or create_default_llm(self.agent.model)
        return create_swarm_workflow(
            llm, 
            self.agent, 
            state_scheme=self.state_scheme, 
            print_messages=None, # disable print_messages in workflow 
            debug=self.debug, 
            agent_map=self.agent_map
        )

    def get_agent(self, name: str, fail_if_not_found: bool = True):
        if name in self.agent_map:
            return self.agent_map[name][0]
        if fail_if_not_found:
            raise ValueError(f"Agent {name} not found")
        return None
    
    def run_stream(self, input: dict, max_turns: int, debug: bool):
        init_len = len(input["messages"])
        for _chunk in self.workflow.stream(
            input=input,
            config={"recursion_limit": max_turns},
            debug=debug or self.debug,
            subgraphs=debug or self.debug,
            ):
            if isinstance(_chunk, Dict):
                for _agent, _resp in _chunk.items():
                    yield {_agent: _resp}
                resp = _resp
            # TODO: parse chunk when workflow debug is enabled and subgraphs is True
            elif isinstance(_chunk, Tuple):
                for _item in _chunk:
                    if isinstance(_item, Dict):
                        for _agent, _resp in _item.items():
                            yield {_agent: _resp}
                        resp = _resp
                    else:
                        yield _item
        agent_name = resp["agent_name"]
        self.agent = self.get_agent(agent_name, False)
        if self.print_messages:
            self.print_messages(resp["messages"][init_len:])
        yield Response(
            messages=resp["messages"][init_len:],
            agent=resp["agent_name"],
            context_variables=resp.get(__CTX_VARS_NAME__, {}),
            handoff=resp.get("handoff", False),
        )
    
    # for now, trying to be compatible with OpenAI Swarm
    def run(
        self,
        messages: List,
        agent: Agent = None,
        context_variables: dict = {},
        stream: bool = False,
        debug: bool = False,
        max_turns: int = 25,
        execute_tools: bool = True,
    ) -> Response:
        if agent:
            self.agent = agent
            # in case you want to create a new workflow with a different agent
            if agent.name not in self.agent_map:
                self.start_agent = agent
                self.agent_map = {}
                self.workflow =self._create_workflow()
        agent_name = self.agent.name
        # TODO: use max_turns instead of Graph's recursion
        init_len = len(messages)
        input={"messages": messages, "agent_name": agent_name, "handoff": True, __CTX_VARS_NAME__: context_variables}
        if stream:
            return self.run_stream(input, max_turns, debug)
        else:
            resp = self.workflow.invoke(
                input=input,
                config={"recursion_limit": max_turns},
                debug=debug or self.debug,
                subgraphs=debug or self.debug,
            )
            agent_name = resp["agent_name"]
            self.agent = self.get_agent(agent_name, False)
            if self.print_messages:
                self.print_messages(resp["messages"][init_len:])
            return Response(
                messages=resp["messages"][init_len:],
                agent=resp["agent_name"],
                context_variables=resp.get(__CTX_VARS_NAME__, {}),
                handoff=resp.get("handoff", False),
            )