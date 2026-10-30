"""Request/response models."""
from typing import Literal

from pydantic import BaseModel, Field

QuestionType = Literal["balanced", "best"]

class Participant(BaseModel):
    language: Literal["en", "it"]
    user_agent: str | None = None
    screen_width: int | None = None
    screen_height: int | None = None
    # demographics

class QuestionKey(BaseModel):
    question_index: int
    question_type: QuestionType

class PlannedQuestion(BaseModel):
    question_index: int
    sample_id: str #? or int Place1 or 1 to chose
    kind: Literal["normal", "repeat"]
    variant_order: list[str]

class ParticipantState(BaseModel):
    participant_id: str
    plan: list[PlannedQuestion]
    answered: list[QuestionKey]

class Answer(BaseModel):
    participant_id: str
    question_index: int
    question_type: QuestionType
    chosen_variant: str
    response_time_ms: int = Field(ge=0)

class UserAdjustment(BaseModel):
    participant_id: str
    question_index: int
    question_type: QuestionType
    base_variant: str   # variant the adjustment was applied on.
    final_t: float = Field(ge=-1, le=1)
    final_q: float = Field(ge=-1, le=1)
    #panel_touched: bool
    confirmed_as_is: bool = False
