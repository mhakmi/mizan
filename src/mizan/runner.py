from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from mizan.assertions import check
from mizan.types import Case, CaseResult, Complete, Errored, ProviderError, Scored, Unscorable


def run_cases(
    cases: Sequence[Case],
    complete: Complete,
    ask_judge: Complete | None,
    concurrency: int,
) -> tuple[CaseResult, ...]:
    def run_one(case: Case) -> CaseResult:
        return run_case(case, complete, ask_judge)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return tuple(pool.map(run_one, cases))


def run_case(case: Case, complete: Complete, ask_judge: Complete | None) -> CaseResult:
    try:
        reply = complete(case.prompt)
    except ProviderError as error:
        return Errored(case_id=case.id, reason=f"The model call failed. {error}")

    checks = []
    try:
        for assertion in case.assertions:
            checks.append(check(assertion, case, reply.text, ask_judge))
    except Unscorable as error:
        return Errored(case_id=case.id, reason=str(error))

    return Scored(
        case_id=case.id,
        output=reply.text,
        prompt_tokens=reply.prompt_tokens,
        checks=tuple(checks),
    )
