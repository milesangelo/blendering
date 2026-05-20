from __future__ import annotations

import pytest
from pydantic import ValidationError

from blendering.schemas import RunOutcome, Verdict


def test_verdict_done_minimal() -> None:
    v = Verdict(status="done", reasoning="Looks good.", confidence=0.9)
    assert v.status == "done"
    assert v.next_step_hint is None


def test_verdict_rejects_invalid_status() -> None:
    with pytest.raises(ValidationError):
        Verdict(status="maybe", reasoning="x")  # type: ignore[arg-type]


def test_verdict_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        Verdict(status="continue", reasoning="x", confidence=1.5)


def test_run_outcome_optional_fields() -> None:
    o = RunOutcome(status="cancelled", iterations=3)
    assert o.last_verdict is None
    assert o.error is None
