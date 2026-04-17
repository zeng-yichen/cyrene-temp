"""Deterministic graders — check tool calls, forbidden tools, output patterns."""

import re

from backend.src.models.eval import EvalTrace, GradeResult


def check_tools_called(trace: EvalTrace, expected_tools: list[str]) -> GradeResult:
    """Check that all expected tools were called."""
    called = {tc.get("name", "") for tc in trace.tool_calls}
    missing = set(expected_tools) - called
    if missing:
        return GradeResult(
            passed=False,
            reason=f"Missing tool calls: {', '.join(missing)}",
            grader="deterministic",
        )
    return GradeResult(passed=True, reason="All expected tools called", grader="deterministic")


def check_forbidden_tools(trace: EvalTrace, forbidden_tools: list[str]) -> GradeResult:
    """Check that no forbidden tools were called."""
    called = {tc.get("name", "") for tc in trace.tool_calls}
    violations = called & set(forbidden_tools)
    if violations:
        return GradeResult(
            passed=False,
            reason=f"Forbidden tool(s) called: {', '.join(violations)}",
            grader="deterministic",
        )
    return GradeResult(passed=True, reason="No forbidden tools called", grader="deterministic")


def check_output_patterns(trace: EvalTrace, assertions: list[str]) -> GradeResult:
    """Check that output matches all assertion patterns."""
    response = trace.user_response or ""
    failures = []
    for assertion in assertions:
        if not re.search(assertion, response, re.IGNORECASE | re.DOTALL):
            failures.append(f"Pattern not found: {assertion}")

    if failures:
        return GradeResult(
            passed=False,
            reason="; ".join(failures),
            grader="deterministic",
        )
    return GradeResult(passed=True, reason="All output assertions matched", grader="deterministic")
