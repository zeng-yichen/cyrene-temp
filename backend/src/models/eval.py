"""Eval framework models — eval framework models."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolArgumentAssertion(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)
    tool: str
    argument_contains: dict[str, Any] | None = None
    argument_matches: dict[str, Any] | None = None


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)
    prompt: str
    expected_tools: list[str] = Field(default_factory=list)
    programmatic_tool_assertions: list[ToolArgumentAssertion] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    output_assertions: list[str] = Field(default_factory=list)
    grading: Literal["strict", "lenient"] = "lenient"
    timeout_seconds: int = 90


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    id: str
    agent: str = ""
    prompt: str = ""
    expected_tools: list[str] = Field(default_factory=list)
    programmatic_tool_assertions: list[ToolArgumentAssertion] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    output_assertions: list[str] = Field(default_factory=list)
    grading: Literal["strict", "lenient"] = "lenient"
    timeout_seconds: int = 60
    type: Literal["single", "conversation"] = "single"
    turns: list[ConversationTurn] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_conversation(self) -> bool:
        return self.type == "conversation" and len(self.turns) > 0


class EvalTrace(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    case_id: str
    trial_number: int = 0
    turn_index: int | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_seconds: float | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    user_response: str | None = None
    error: str | None = None


class GradeResult(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    passed: bool
    reason: str
    grader: Literal["deterministic", "llm"]
    rubric_scores: dict[str, bool] | None = None


class TurnResult(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)
    turn_index: int
    prompt: str
    trace: EvalTrace | None = None
    grade_results: list[GradeResult] = Field(default_factory=list)
    passed: bool = False
    failures: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)
    case: EvalCase
    traces: list[EvalTrace] = Field(default_factory=list)
    grade_results: list[GradeResult] = Field(default_factory=list)
    passed: bool = False
    pass_count: int = 0
    failures: list[str] = Field(default_factory=list)
    turn_results: list[TurnResult] = Field(default_factory=list)


class EvalRun(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)
    run_id: str
    start_time: datetime
    end_time: datetime | None = None
    results: list[EvalResult] = Field(default_factory=list)

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def passed_cases(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed_cases / self.total_cases if self.total_cases else 0.0
