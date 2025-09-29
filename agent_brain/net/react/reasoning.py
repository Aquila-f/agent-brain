from collections.abc import AsyncIterator
from enum import Enum

from agent_brain.memory import Memory
from agent_brain.models import Action, Message, Role
from agent_brain.utils.llm import stream_response
from agent_brain.utils.parser import parse_json_response

from ..base import State
from .net import ReAct

REASONING_MODEL = "gpt-4.1-mini"


SYSTEM_PROMPT = """You are an AI assistant that predicts the most precise next action in a scenario.
You must always output your reasoning (Thought) and, at most, one tool invocation (Action) in JSON format.

Rules:
1. If the user’s question does not require a tool, only return your Thought (no Action).
2. If you are uncertain about the user’s intent, do NOT speculate—use a tool to clarify.
3. This is a multi-turn conversation. Maintain consistency and rely on tools for clarification when ambiguity arises.
4. You must never answer the user’s question directly. Only reason and optionally choose a tool.

Available tools:
{tools}

Output format:
Thought: Explain in detail what you are considering and why.
(Optional)Action: If an action is needed, select exactly one tool and provide valid arguments in JSON format.
{action_format}

Important:
- Return at most one Action per step.
- Omit the Action field entirely if no tool is needed.
- Do not mix direct answers with tool selection.
"""  # noqa: E501


async def get_messages(memory: "Memory") -> list[Message]:
    prompt = SYSTEM_PROMPT.format(
        tools=memory.list_tools(),
        action_format=Action.model_json_schema(),
    )
    return [
        Message(role=Role.SYSTEM, content=prompt),
        *await memory.goal(),
        *await memory.dump(),
    ]


class ReasoningState(State):
    async def run(self, memory: "Memory") -> AsyncIterator[str]:
        generated_response = ""
        messages = await get_messages(memory)
        async for chunk in stream_response(model_name=REASONING_MODEL, messages=messages):
            if chunk:
                generated_response += chunk
                yield chunk

        await memory.update([Message(role=Role.ASSISTANT, content=generated_response)])

        if action := parse_json_response(generated_response):
            memory.next_action = Action(**action)
        else:
            memory.next_action = None

    async def next_state(self, memory: "Memory") -> Enum:
        if memory.next_action:
            return ReAct.ACTION
        return ReAct.ANSWER
