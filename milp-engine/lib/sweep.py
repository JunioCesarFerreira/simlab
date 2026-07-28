"""
MILP parameter sweep: solve the model over a cartesian parameter grid,
deduplicate solutions by genotype and report every solve for auditability.

Ported from the checkpoint/dedup pattern of wsn-milp-nsga3-p2/milp/runner.py,
with two deliberate changes:
  - genotype bit i follows problem["candidates"][i] order (ChromosomeP2.mask
    convention), never a name sort;
  - persistence is injected via callbacks (on_record / should_stop) so this
    module stays MongoDB-free and unit-testable. The engine (phase 3) wires
    these to the milp_sweeps collection.
"""
import itertools
import logging
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

from .models.base import MilpModel
from .solver.base import SolverBackend

log = logging.getLogger(__name__)


@dataclass(slots=True)
class SolveRecord:
    """Outcome of one grid point. Emitted for every combination, including
    duplicates and failures — the params→topology mapping is itself a result."""
    index: int
    params: dict[str, float]
    status: str                      # solver status, or "DUPLICATE"-flagged success
    genotype: Optional[str] = None
    mask: Optional[list[int]] = None
    n_installed: Optional[int] = None
    obj_value: Optional[float] = None
    mip_gap: Optional[float] = None
    runtime_s: float = 0.0
    is_duplicate: bool = False
    message: str = ""


@dataclass(slots=True)
class SweepResult:
    records: list[SolveRecord] = field(default_factory=list)
    unique_masks: dict[str, list[int]] = field(default_factory=dict)  # genotype -> mask
    cancelled: bool = False

    @property
    def n_solved(self) -> int:
        return sum(1 for r in self.records if r.genotype is not None)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.records if r.genotype is None)


def expand_grid(
    grid: Mapping[str, Sequence[float]],
    fixed_params: Optional[Mapping[str, float]] = None,
) -> list[dict[str, float]]:
    """
    Cartesian product of the sweep grid merged with fixed parameters.

    Deterministic: grid keys are iterated in sorted order, values in the
    given order, so combination index i is stable across runs — this is what
    makes checkpoint resumption (start_index) safe.
    """
    fixed = dict(fixed_params or {})
    overlap = set(grid) & set(fixed)
    if overlap:
        raise ValueError(f"Parameters both swept and fixed: {sorted(overlap)}")
    keys = sorted(grid)
    for k in keys:
        if not grid[k]:
            raise ValueError(f"Empty value list for swept parameter '{k}'.")
    combos: list[dict[str, float]] = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combo = dict(fixed)
        combo.update({k: float(v) for k, v in zip(keys, values)})
        combos.append(combo)
    return combos


def run_sweep(
    model: MilpModel,
    problem: Mapping,
    grid: Mapping[str, Sequence[float]],
    backend: SolverBackend,
    fixed_params: Optional[Mapping[str, float]] = None,
    time_limit_s: Optional[float] = None,
    mip_gap: Optional[float] = None,
    start_index: int = 0,
    seen_genotypes: Optional[set[str]] = None,
    on_record: Optional[Callable[[SolveRecord], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> SweepResult:
    """
    Solve every grid combination, deduplicating by genotype.

    start_index / seen_genotypes resume an interrupted sweep (checkpoint);
    on_record is invoked after each combination for incremental persistence;
    should_stop is polled before each solve for cooperative cancellation.
    """
    if time_limit_s is None:
        time_limit_s = model.solver_defaults.get("time_limit_s")
    if mip_gap is None:
        mip_gap = model.solver_defaults.get("mip_gap")

    combos = expand_grid(grid, fixed_params)
    seen = set(seen_genotypes or ())
    result = SweepResult()

    log.info(
        "[sweep] model=%s backend=%s combos=%d (resuming at %d, %d genotypes known)",
        model.key, backend.name, len(combos), start_index, len(seen),
    )

    for index in range(start_index, len(combos)):
        if should_stop is not None and should_stop():
            result.cancelled = True
            log.info("[sweep] cancelled at combination %d/%d", index, len(combos))
            break

        params = combos[index]
        try:
            built = model.build(problem, params)
            solve = backend.solve(built.ir, time_limit_s=time_limit_s, mip_gap=mip_gap)
        except Exception as exc:
            log.exception("[sweep] combo %d failed to build/solve", index)
            record = SolveRecord(
                index=index, params=params, status="ERROR", message=str(exc)
            )
            result.records.append(record)
            if on_record is not None:
                on_record(record)
            continue

        if solve.has_solution:
            mask = built.decode_mask(solve.values)
            genotype = "".join(str(bit) for bit in mask)
            duplicate = genotype in seen
            if not duplicate:
                seen.add(genotype)
                result.unique_masks[genotype] = mask
            record = SolveRecord(
                index=index,
                params=params,
                status=solve.status,
                genotype=genotype,
                mask=mask,
                n_installed=sum(mask),
                obj_value=solve.obj_value,
                mip_gap=solve.mip_gap,
                runtime_s=solve.runtime_s,
                is_duplicate=duplicate,
            )
        else:
            record = SolveRecord(
                index=index,
                params=params,
                status=solve.status,
                runtime_s=solve.runtime_s,
                message=solve.message,
            )

        result.records.append(record)
        if on_record is not None:
            on_record(record)

    log.info(
        "[sweep] done: %d records, %d unique genotypes, %d without solution",
        len(result.records), len(result.unique_masks), result.n_failed,
    )
    return result
