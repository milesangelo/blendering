"""Verifier tests using synthetic scene snapshots."""

from __future__ import annotations

from blendering.config import VerifierConfig
from blendering.schemas import PartSpec, Plan, PositionSpec
from blendering.verifier import verify


def _abs_part(
    pid: str,
    primitive: str,
    dims: dict[str, float],
    xyz: tuple[float, float, float],
) -> PartSpec:
    return PartSpec(
        id=pid,
        description=pid,
        primitive=primitive,  # type: ignore[arg-type]
        dimensions=dims,
        position=PositionSpec(mode="absolute", xyz=xyz),
    )


def _rel_part(
    pid: str,
    primitive: str,
    dims: dict[str, float],
    anchor: str,
    face: str,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PartSpec:
    return PartSpec(
        id=pid,
        description=pid,
        primitive=primitive,  # type: ignore[arg-type]
        dimensions=dims,
        position=PositionSpec(
            mode="relative",
            anchor_part=anchor,
            anchor_face=face,  # type: ignore[arg-type]
            offset=offset,
        ),
    )


def _cube_snapshot(name: str, center: tuple[float, float, float], size: float) -> dict:
    cx, cy, cz = center
    h = size / 2.0
    return {
        "primitive_guess": "cube",
        "vert_count": 8,
        "world_location": [cx, cy, cz],
        "world_bbox_min": [cx - h, cy - h, cz - h],
        "world_bbox_max": [cx + h, cy + h, cz + h],
        "rotation_euler_deg": [0.0, 0.0, 0.0],
    }


def _cfg() -> VerifierConfig:
    return VerifierConfig()


def test_all_parts_ok() -> None:
    plan = Plan(goal="x", parts=[_abs_part("a", "cube", {"x": 1.0, "y": 1.0, "z": 1.0}, (0.0, 0.0, 0.0))])
    snapshot = {"objects": {"a": _cube_snapshot("a", (0.0, 0.0, 0.0), 1.0)}}
    diff = verify(plan, snapshot, _cfg())
    assert [p.status for p in diff.parts] == ["ok"]
    assert diff.extras == []
    assert diff.is_structural is False


def test_missing_part_is_structural_by_default() -> None:
    plan = Plan(goal="x", parts=[_abs_part("a", "cube", {"x": 1.0, "y": 1.0, "z": 1.0}, (0.0, 0.0, 0.0))])
    snapshot: dict = {"objects": {}}
    diff = verify(plan, snapshot, _cfg())
    assert [p.status for p in diff.parts] == ["missing"]
    assert diff.is_structural is True


def test_dimension_off_outside_tolerance() -> None:
    plan = Plan(goal="x", parts=[_abs_part("a", "cube", {"x": 1.0, "y": 1.0, "z": 1.0}, (0.0, 0.0, 0.0))])
    snapshot = {"objects": {"a": _cube_snapshot("a", (0.0, 0.0, 0.0), 1.5)}}  # 50% larger
    diff = verify(plan, snapshot, _cfg())
    assert diff.parts[0].status == "off"
    assert any("x" in i or "dimension" in i.lower() for i in diff.parts[0].issues)


def test_position_off_outside_tolerance() -> None:
    plan = Plan(goal="x", parts=[_abs_part("a", "cube", {"x": 1.0, "y": 1.0, "z": 1.0}, (0.0, 0.0, 0.0))])
    snapshot = {"objects": {"a": _cube_snapshot("a", (0.5, 0.0, 0.0), 1.0)}}  # 50cm off, tol is 10cm
    diff = verify(plan, snapshot, _cfg())
    assert diff.parts[0].status == "off"
    assert any("position" in i.lower() for i in diff.parts[0].issues)


def test_relative_position_uses_anchor_top_face() -> None:
    plan = Plan(
        goal="lamp on table",
        parts=[
            _abs_part("table", "cube", {"x": 1.0, "y": 1.0, "z": 1.0}, (0.0, 0.0, 0.0)),
            # lamp_base sits on top of the table (z = 0.5 = table top)
            _rel_part(
                "lamp_base", "cube",
                {"x": 0.2, "y": 0.2, "z": 0.1},
                anchor="table", face="top",
            ),
        ],
    )
    # Place lamp_base centered just above the table top: center at (0, 0, 0.55) = table_top + half-height
    snapshot = {
        "objects": {
            "table": _cube_snapshot("table", (0.0, 0.0, 0.0), 1.0),
            "lamp_base": _cube_snapshot("lamp_base", (0.0, 0.0, 0.55), 0.1),
        }
    }
    diff = verify(plan, snapshot, _cfg())
    statuses = {p.part_id: p.status for p in diff.parts}
    assert statuses == {"table": "ok", "lamp_base": "ok"}


def test_extras_listed() -> None:
    plan = Plan(goal="x", parts=[_abs_part("a", "cube", {"x": 1.0, "y": 1.0, "z": 1.0}, (0.0, 0.0, 0.0))])
    snapshot = {
        "objects": {
            "a": _cube_snapshot("a", (0.0, 0.0, 0.0), 1.0),
            "StrayCube": _cube_snapshot("StrayCube", (5.0, 0.0, 0.0), 1.0),
        }
    }
    diff = verify(plan, snapshot, _cfg())
    assert "StrayCube" in diff.extras


def test_structural_when_two_parts_off() -> None:
    plan = Plan(
        goal="x",
        parts=[
            _abs_part("a", "cube", {"x": 1.0, "y": 1.0, "z": 1.0}, (0.0, 0.0, 0.0)),
            _abs_part("b", "cube", {"x": 1.0, "y": 1.0, "z": 1.0}, (2.0, 0.0, 0.0)),
        ],
    )
    snapshot = {
        "objects": {
            "a": _cube_snapshot("a", (1.0, 0.0, 0.0), 1.0),  # off
            "b": _cube_snapshot("b", (5.0, 0.0, 0.0), 1.0),  # off
        }
    }
    diff = verify(plan, snapshot, _cfg())
    assert diff.is_structural is True
