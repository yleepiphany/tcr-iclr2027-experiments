"""Fixed, nested row budgets for the π0.5 A_new→B_new appendix controls.

The existing cap-16 selector defines the shared 1× B row identities.  Low takes
eight of those rows per request/module; high retains all sixteen and adds eight
new rows.  Time projections have only one available row per request, so their
realized count is unchanged at all three tiers and must be disclosed.
"""
from __future__ import annotations

from pi05_table3_contract import FLOWS, row_indices


TIERS = {"low": 8, "base": 16, "high": 24}
TIME_MODULES = frozenset({"model.time_mlp_in", "model.time_mlp_out"})
EXPECTED_TOTALS = {"low": 666_000, "base": 1_331_600, "high": 1_997_200}
EXPECTED_SINGLE_UNION = 5_772_800


def _spaced(length: int, count: int) -> list[int]:
    if length < count or count < 1:
        raise ValueError("not enough unique rows")
    if count == 1:
        return [0]
    return [(length - 1) * index // (count - 1) for index in range(count)]


def nested_row_indices(
    nrows: int, tier: str, flow: int, request_slot: int,
    *, cameras: int = 1, camera: int = 0,
) -> list[int]:
    """Return one flow/view slice, preserving the original cap-16 allocation.

    Row identity is (request, flow, camera, token); new high rows cannot replace
    base rows.  For the scalar time projections all tiers have the same one row.
    """
    if tier not in TIERS or flow not in FLOWS or not 0 <= request_slot < 5:
        raise ValueError("unregistered budget tier, flow, or request slot")
    if nrows < 1 or cameras < 1 or not 0 <= camera < cameras:
        raise ValueError("invalid row/view geometry")
    capacity = nrows * cameras
    if 1 < capacity < 24:
        raise ValueError("module cannot realize the registered high tier")
    if capacity == 1:
        return row_indices(nrows, 16, flow, request_slot,
                           cameras=cameras, camera=camera)

    base_tokens = _spaced(capacity, 16)
    base_positions = range(16) if tier != "low" else range(0, 16, 2)
    entries = [(base_tokens[index], FLOWS[(index + request_slot) % 3])
               for index in base_positions]
    if tier == "high":
        base_set = set(base_tokens)
        available = [token for token in _spaced(capacity, 24)
                     if token not in base_set]
        if len(available) < 8:
            available = [token for token in range(capacity)
                         if token not in base_set]
        extra = [available[index] for index in _spaced(len(available), 8)]
        entries.extend((token, FLOWS[(16 + index + request_slot) % 3])
                       for index, token in enumerate(extra))
    selected = [token % nrows for token, stage in entries
                if stage == flow and token // nrows == camera]
    if tier == "base":
        original = row_indices(nrows, 16, flow, request_slot,
                               cameras=cameras, camera=camera)
        if selected != original:
            raise AssertionError("base tier changed the frozen cap-16 rows")
    return selected


def realized_budget(module_names: set[str] | list[str] | tuple[str, ...],
                    tier: str, *, experts: int = 4, requests: int = 50) -> int:
    """Full four-expert B budget, including two non-scalable time modules."""
    if tier not in TIERS or experts < 1 or requests < 1:
        raise ValueError("invalid budget parameters")
    names = set(module_names)
    if len(names) != 418 or not TIME_MODULES <= names:
        raise ValueError("budget requires the exact 418-module full scope")
    return experts * requests * (len(names - TIME_MODULES) * TIERS[tier]
                                 + len(TIME_MODULES))


def single_union_budget(module_names: set[str] | list[str] | tuple[str, ...],
                        *, experts: int = 4, requests: int = 50) -> int:
    """A∪B single-solve budget: original A cap-10 plus B cap-16 rows."""
    names = set(module_names)
    if len(names) != 418 or not TIME_MODULES <= names:
        raise ValueError("budget requires the exact 418-module full scope")
    vision = {name for name in names if ".vision_tower." in name}
    if len(vision) != 162:
        raise ValueError("vision module scope differs")
    other = names - vision - TIME_MODULES
    per_expert = (len(vision) * (requests * 3 * 3 * 10 + requests * 16)
                  + len(other) * (requests * 3 * 10 + requests * 16)
                  + len(TIME_MODULES) * (requests * 3 + requests))
    return experts * per_expert
