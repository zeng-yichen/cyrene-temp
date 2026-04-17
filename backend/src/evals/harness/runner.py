"""Eval harness runner — runs eval cases against agents."""

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from backend.src.models.eval import EvalCase, EvalResult, EvalRun, EvalTrace, GradeResult

logger = logging.getLogger(__name__)


def load_cases(cases_dir: str | Path) -> list[EvalCase]:
    """Load eval cases from YAML files in a directory."""
    cases_path = Path(cases_dir)
    cases = []
    for yaml_file in sorted(cases_path.glob("*.yaml")) + sorted(cases_path.glob("*.yml")):
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, list):
                cases.extend(EvalCase.model_validate(d) for d in data)
            elif isinstance(data, dict):
                cases.append(EvalCase.model_validate(data))
        except Exception as e:
            logger.warning("Failed to load eval case %s: %s", yaml_file, e)
    return cases


def run_case(case: EvalCase) -> EvalResult:
    """Run a single eval case against its target agent."""
    start = datetime.now(timezone.utc)
    trace = EvalTrace(
        case_id=case.id,
        trial_number=0,
        start_time=start,
    )

    try:
        response = _execute_agent(case)
        trace.user_response = response
        trace.end_time = datetime.now(timezone.utc)
        trace.duration_seconds = (trace.end_time - start).total_seconds()
    except Exception as e:
        trace.error = str(e)
        trace.end_time = datetime.now(timezone.utc)
        trace.duration_seconds = (trace.end_time - start).total_seconds()

    grade_results = _grade_trace(trace, case)
    passed = all(g.passed for g in grade_results) if grade_results else trace.error is None

    return EvalResult(
        case=case,
        traces=[trace],
        grade_results=grade_results,
        passed=passed,
        pass_count=1 if passed else 0,
        failures=[g.reason for g in grade_results if not g.passed],
    )


def run_eval(cases_dir: str | Path, agent_filter: str | None = None) -> EvalRun:
    """Run all eval cases and return an EvalRun summary."""
    cases = load_cases(cases_dir)
    if agent_filter:
        cases = [c for c in cases if c.agent == agent_filter]

    run = EvalRun(
        run_id=str(uuid.uuid4()),
        start_time=datetime.now(timezone.utc),
    )

    for case in cases:
        logger.info("Running eval case: %s", case.id)
        result = run_case(case)
        run.results.append(result)
        status = "PASS" if result.passed else "FAIL"
        logger.info("  %s: %s", case.id, status)

    run.end_time = datetime.now(timezone.utc)
    logger.info(
        "Eval run complete: %d/%d passed (%.0f%%)",
        run.passed_cases,
        run.total_cases,
        run.pass_rate * 100,
    )
    return run


def _execute_agent(case: EvalCase) -> str:
    """Dispatch to the correct agent based on case.agent field."""
    agent_name = case.agent.lower()

    if agent_name == "stelle":
        from backend.src.agents.stelle_adapter import run_stelle
        return run_stelle(case.context.get("company", "test"), case.prompt) or ""

    elif agent_name == "aglaea":
        from backend.src.agents.aglaea_adapter import run_aglaea
        return run_aglaea(
            case.context.get("client_name", "Test Client"),
            case.context.get("company", "test"),
        ) or ""

    elif agent_name == "cyrene":
        from backend.src.agents.demiurge import Cyrene
        cyrene = Cyrene()
        result = cyrene.rewrite_single_post(
            post_text=case.context.get("post_text", case.prompt),
            style_instruction=case.context.get("style", ""),
        )
        return result.get("final_post", "")

    elif agent_name == "castorice":
        from backend.src.agents.castorice import Castorice
        castorice = Castorice()
        return castorice.fact_check_post(case.prompt, case.context.get("company", "test"))

    elif agent_name == "herta":
        from backend.src.agents.herta_adapter import run_herta
        return run_herta(case.context.get("company", "test")) or ""

    else:
        raise ValueError(f"Unknown agent: {case.agent}")


def _grade_trace(trace: EvalTrace, case: EvalCase) -> list[GradeResult]:
    """Apply all graders to a trace."""
    results = []

    if trace.error:
        results.append(GradeResult(passed=False, reason=f"Agent error: {trace.error}", grader="deterministic"))
        return results

    from backend.src.evals.graders.deterministic import check_tools_called, check_forbidden_tools, check_output_patterns

    if case.expected_tools:
        results.append(check_tools_called(trace, case.expected_tools))

    if case.forbidden_tools:
        results.append(check_forbidden_tools(trace, case.forbidden_tools))

    if case.output_assertions:
        results.append(check_output_patterns(trace, case.output_assertions))

    return results
