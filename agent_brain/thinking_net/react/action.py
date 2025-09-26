from collections.abc import AsyncIterator
from enum import Enum

from agent_brain.memory import Memory
from agent_brain.models import Message, Role

from ..base import State
from .net import ReAct


class ActionState(State):
    async def run(self, memory: "Memory") -> AsyncIterator[str]:
        if not memory.next_action:
            return

        if tool := memory.get_tool(memory.next_action.name):
            result = await tool.execute(**memory.next_action.args)
            await memory.update([Message(role=Role.ACT, content=str(result))])
        yield "\n"

    async def next_state(self, memory: "Memory") -> Enum:
        return ReAct.REASONING
