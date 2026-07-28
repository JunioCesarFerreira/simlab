"""
Handoff: turn a completed MILP sweep into a batch experiment.

The generated experiment document is exactly what the GUI's LaunchWizard
produces for strategy="batch": the mo-engine's BatchStrategy picks it up via
change stream and the whole existing pipeline (master-node, Cooja workers,
Pareto analysis, GUI) runs unchanged. The milp-engine never talks to Cooja.
"""
from datetime import datetime
from typing import Any, Mapping

# Statuses duplicated from pylib.db.EnumStatus values to keep this module
# importable without pylib in unit tests; the runner passes the real enum.
_WAITING = "Waiting"

_VALID_MACS = (0, 1)  # 0 = CSMA/CA, 1 = TSCH (ChromosomeBase convention)


def build_batch_experiment(
    sweep: Mapping[str, Any],
    unique_masks: Mapping[str, list[int]],
) -> dict[str, Any]:
    """
    Build the batch-experiment document for the unique topologies of a sweep.

    Raises ValueError when the sweep's batch_options cannot produce a valid
    batch experiment (no topologies, missing objectives, bad MAC list).
    """
    if not unique_masks:
        raise ValueError("Sweep produced no feasible topologies; nothing to hand off.")

    batch_options: dict[str, Any] = dict(sweep.get("batch_options") or {})

    objectives = batch_options.get("objectives") or []
    if not objectives:
        raise ValueError(
            "batch_options.objectives is required to create the batch experiment."
        )

    mac_protocols = batch_options.get("mac_protocols") or [0]
    if not all(m in _VALID_MACS for m in mac_protocols):
        raise ValueError(f"Invalid mac_protocols {mac_protocols}; allowed: {list(_VALID_MACS)}.")

    simulation = dict(batch_options.get("simulation") or {})
    simulation.setdefault("duration", 120)

    # The MILP does not decide the MAC gene: each unique topology is emitted
    # once per requested MAC protocol. Genotype order is deterministic.
    chromosomes = [
        {"mac_protocol": mac, "mask": list(unique_masks[genotype])}
        for genotype in sorted(unique_masks)
        for mac in mac_protocols
    ]

    name = batch_options.get("experiment_name") or f"{sweep.get('name', 'MILP sweep')} — batch"

    return {
        "name": name,
        "status": _WAITING,
        "system_message": None,
        "created_time": datetime.now(),
        "start_time": None,
        "end_time": None,
        "parameters": {
            "strategy": "batch",
            "algorithm": {},
            "simulation": simulation,
            "problem": sweep["problem"],
            "objectives": objectives,
            "chromosomes": chromosomes,
            "random_seed": batch_options.get("random_seed", 42),
        },
        "source_repository_options": batch_options.get("source_repository_options", {}),
        "data_conversion_config": batch_options.get(
            "data_conversion_config",
            {"node_col": "node", "time_col": "time", "metrics": []},
        ),
        "pareto_front": None,
        # Provenance: lets the GUI badge experiments born from a MILP sweep
        "milp_sweep_id": sweep.get("_id"),
    }
