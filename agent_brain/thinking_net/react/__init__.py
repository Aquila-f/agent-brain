from enum import Enum, auto


class StateType(Enum):
    REASONING = auto()
    ACTION = auto()
    ANSWER = auto()


from ..base import State  # noqa: E402
from .action import ActionState  # noqa: E402
from .answer import AnswerState  # noqa: E402
from .reasoning import ReasoningState  # noqa: E402


def create_states() -> dict[Enum, State]:
    return {
        StateType.REASONING: ReasoningState(),
        StateType.ACTION: ActionState(),
        StateType.ANSWER: AnswerState(),
    }


__all__ = ["create_states"]
