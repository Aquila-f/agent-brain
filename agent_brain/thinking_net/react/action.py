from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from agent_brain.models import Message, Role

from ..base import State
from . import StateType

if TYPE_CHECKING:
    from agent_brain.memory import MemoryEnv


class ActionState(State):
    async def _run(self, env: "MemoryEnv") -> AsyncIterator[str]:
        if env.next_action:
            result = await env.act()
            env.history.append(Message(role=Role.USER, content=str(result)))
        yield "\n"

    async def _transition(self, env: "MemoryEnv") -> None:
        env.set_state(StateType.REASONING)
