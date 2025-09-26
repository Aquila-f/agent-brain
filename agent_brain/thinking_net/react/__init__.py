from enum import Enum

from ..base import State
from .net import ReAct


def create_net() -> dict[Enum, State]:
    from .action import ActionState
    from .answer import AnswerState
    from .reasoning import ReasoningState

    return {
        ReAct.REASONING: ReasoningState(),
        ReAct.ACTION: ActionState(),
        ReAct.ANSWER: AnswerState(),
    }


__all__ = ["create_net"]
