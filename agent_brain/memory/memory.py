from collections.abc import AsyncIterator
from enum import Enum

from agent_brain.models import Action, Message, Role
from agent_brain.thinking_net.base import State
from agent_brain.tool import BaseTool


class MemoryEnv:
    def __init__(self, states: dict[Enum, State], tools: list[BaseTool]) -> None:
        self.history: list[Message] = []
        self.done: bool = False
        self.next_action: Action | None = None
        self.state: State = list(states.values())[0]
        self.state_map: dict[Enum, State] = states
        self.available_tools: dict[str, BaseTool] = {tool.name: tool for tool in tools}

    async def step(self) -> AsyncIterator[str]:
        async for output in self.state.run(self):
            yield output

    def set_state(self, next_type: Enum):
        self.state.on_exit(self)
        self.state = self.state_map[next_type]
        self.state.on_enter(self)

    async def act(self) -> str:
        if self.next_action:
            action = self.next_action
            if tool := self.available_tools.get(action.name):
                result = await tool.execute(**action.args)
                return result

            return f"Tool '{action.name}' not found."
        return "No action to perform."

    def list_tools(self) -> str:
        return ",".join([str(tool.to_dict()) for tool in self.available_tools.values()])

    def start_task(self, task: str) -> None:
        self.history.append(Message(role=Role.USER, content=task))
        self.done = False
        self.next_action = None
