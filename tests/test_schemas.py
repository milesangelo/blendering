from __future__ import annotations

import pytest
from pydantic import ValidationError

from blendering.schemas import RunOutcome, Verdict
from blendering.schemas import (
    PartProposal,
    PartSpec,
    Plan,
    PositionSpec,
)


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


def test_position_spec_absolute() -> None:
    p = PositionSpec(mode="absolute", xyz=(0.0, 0.0, 0.5))
    assert p.mode == "absolute"
    assert p.xyz == (0.0, 0.0, 0.5)
    assert p.anchor_part is None


def test_position_spec_relative() -> None:
    p = PositionSpec(
        mode="relative",
        anchor_part="table",
        anchor_face="top",
        offset=(0.0, 0.0, 0.0),
    )
    assert p.anchor_part == "table"
    assert p.anchor_face == "top"


def test_part_spec_minimum_valid() -> None:
    part = PartSpec(
        id="lamp_base",
        description="cylindrical wooden lamp base",
        primitive="cylinder",
        dimensions={"radius": 0.15, "height": 0.05},
        position=PositionSpec(mode="absolute", xyz=(0.0, 0.0, 0.0)),
    )
    assert part.id == "lamp_base"
    assert part.orientation_deg == (0.0, 0.0, 0.0)


def test_part_spec_rejects_unknown_primitive() -> None:
    with pytest.raises(ValidationError):
        PartSpec(
            id="x",
            description="x",
            primitive="dodecahedron",  # not in the Literal
            dimensions={},
            position=PositionSpec(mode="absolute", xyz=(0.0, 0.0, 0.0)),
        )


def test_plan_round_trip_json() -> None:
    plan = Plan(
        goal="lamp on table",
        parts=[
            PartSpec(
                id="table",
                description="oak table",
                primitive="cube",
                dimensions={"x": 1.0, "y": 0.6, "z": 0.05},
                position=PositionSpec(mode="absolute", xyz=(0.0, 0.0, 0.75)),
            ),
            PartSpec(
                id="lamp_base",
                description="lamp base",
                primitive="cylinder",
                dimensions={"radius": 0.1, "height": 0.05},
                position=PositionSpec(
                    mode="relative",
                    anchor_part="table",
                    anchor_face="top",
                    offset=(0.0, 0.0, 0.0),
                ),
            ),
        ],
    )
    blob = plan.model_dump_json()
    restored = Plan.model_validate_json(blob)
    assert restored == plan
    assert restored.version == 1


def test_part_proposal_minimal() -> None:
    pp = PartProposal(description="a book on the table")
    assert pp.description == "a book on the table"
    assert pp.rationale is None
