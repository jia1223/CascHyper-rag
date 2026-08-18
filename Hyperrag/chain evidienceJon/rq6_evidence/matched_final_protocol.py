"""Shared contract for RQ6's controlled generator-visible evidence comparison."""

from __future__ import annotations

from typing import Any, Callable


MATCHED_FINAL_CONTEXT_PROTOCOL = "rq6_matched_final_source_evidence_v3"
MATCHED_FINAL_SOURCE_TOKEN_BUDGET = 12_000


def select_complete_source_units(
    candidates: list[dict[str, Any]],
    token_budget: int,
    token_count: Callable[[str], int],
) -> tuple[list[dict[str, Any]], int]:
    """Keep an ordered prefix of whole source units within a shared budget.

    The first unit that would overflow ends the sequence.  Later candidates are
    intentionally not considered: skipping it would change the declared native
    evidence order to optimise coverage after seeing the budget.
    """
    selected: list[dict[str, Any]] = []
    used = 0
    for candidate in candidates:
        tokens = int(token_count(str(candidate["text"])))
        if used + tokens > token_budget:
            break
        selected.append({**candidate, "source_token_count": tokens})
        used += tokens
    return selected, used
