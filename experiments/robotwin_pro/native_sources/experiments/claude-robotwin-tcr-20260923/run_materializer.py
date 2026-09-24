#!/usr/bin/env python3
"""Run the generic full-418 PI0.5 materializer under the RoboTwin M=3 contract."""
from __future__ import annotations

import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
VLA = HERE.parents[1]
SOURCE = VLA.parent / "pi05_lora_finetune_v2_20260826"
sys.path[:0] = [str(HERE), str(VLA / "experiments/claude-20260917"),
                str(VLA / "scripts"), str(VLA / "src"), str(SOURCE / "src")]

import robotwin_solver_contract as contract  # noqa: E402
import materialize_second_round_v3 as materializer  # noqa: E402
import pi05_table3_contract as table3_contract  # noqa: E402


def main() -> None:
    which = os.environ.get("ROBOTWIN_TCR_PASS")
    if which not in {"A", "B"}:
        raise ValueError("ROBOTWIN_TCR_PASS must be A or B")
    # The generic solver imports this module from inside main().
    sys.modules["pi05_tcr_e_dense_contract"] = contract
    materializer.validate_second_round_config = contract.validate_second_round_config
    materializer.validate_second_round_trace = contract.validate_second_round_trace
    materializer.SECOND_ROUND_MASSES = {"uniform_three": "none"}
    # The generic pass-B branch imports Table-3's cap-16 row validator inside
    # main().  Keep its deterministic row_indices helper, but replace the
    # four-expert/50-row validator with this experiment's M=3/150-row contract.
    table3_contract.validate_rows = (
        lambda metrics, names, variant: contract.validate_realized_rows(metrics)
    )
    materializer.main()


if __name__ == "__main__":
    main()
