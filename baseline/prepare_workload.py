#!/usr/bin/env python3
"""
Generates a workload of simulation bundles for the script-based baseline,
matching the ``population_size = 30, fixed_motes = 50`` workload described
in §5.2 of the SimLab paper.

Each bundle is a directory under ``inputs/sim_XXX/`` containing:
  - ``simulation.csc`` — Cooja configuration with 50 randomly placed motes
  - ``positions.dat``  — companion file with mote positions
  - any firmware files provided via ``--firmware-dir``

Usage:
  python prepare_workload.py \
      --template ../mo-engine/simulation_template.xml \
      --out      ./inputs \
      --count    30 \
      --motes    50 \
      --region  -150 -150 150 150 \
      --duration 180 \
      --seed    42

The intent is NOT to compete with the mo-engine pipeline.  This script
exists solely so the baseline orchestrator has reproducible input bundles,
independent of MongoDB or GridFS.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

# Reuse the same Cooja file generator the mo-engine uses, so the format
# stays consistent.  Requires SIMLAB root on PYTHONPATH.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pylib import cooja_files
from pylib.config.simulator import SimulationConfig


def random_simulation_config(
    sim_index: int,
    motes: int,
    region: tuple[float, float, float, float],
    duration: int,
    seed: int,
    radius_of_reach: float,
    radius_of_inter: float,
) -> SimulationConfig:
    rng = random.Random(seed)
    xmin, ymin, xmax, ymax = region
    fixed_motes = []
    for i in range(motes):
        fixed_motes.append({
            "name": f"node{i}",
            "source_code": "node.c",
            "position": [
                round(rng.uniform(xmin, xmax), 2),
                round(rng.uniform(ymin, ymax), 2),
            ],
        })
    cfg: SimulationConfig = {
        "name": f"baseline-sim-{sim_index:03d}",
        "duration": duration,
        "randomSeed": seed,
        "radiusOfReach": radius_of_reach,
        "radiusOfInter": radius_of_inter,
        "region": list(region),
        "simulationElements": {
            "fixedMotes": fixed_motes,
            "mobileMotes": [],
        },
    }
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default=str(ROOT / "mo-engine" / "simulation_template.xml"),
                        help="Path to the Cooja .csc template (default: mo-engine/simulation_template.xml)")
    parser.add_argument("--out", default="./inputs",
                        help="Directory where sim_XXX bundles will be written")
    parser.add_argument("--count", type=int, default=30,
                        help="Number of simulation bundles to generate (default: 30, per §5.2)")
    parser.add_argument("--motes", type=int, default=50,
                        help="Number of fixed motes per simulation (default: 50, per §5.2)")
    parser.add_argument("--region", type=float, nargs=4, default=(-150.0, -150.0, 150.0, 150.0),
                        metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    parser.add_argument("--duration", type=int, default=180,
                        help="Simulated duration in seconds (default: 180)")
    parser.add_argument("--radius-of-reach", type=float, default=50.0)
    parser.add_argument("--radius-of-inter", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=42,
                        help="Base RNG seed; bundles use seed + sim_index")
    parser.add_argument("--firmware-dir", default=None,
                        help="Optional directory whose files are copied into each bundle "
                             "(e.g. compiled .c sources to be uploaded to Cooja)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    firmware_files: list[Path] = []
    if args.firmware_dir:
        firmware_files = [p for p in Path(args.firmware_dir).iterdir() if p.is_file()]
        print(f"[prepare] {len(firmware_files)} firmware file(s) will be copied into each bundle.")

    for i in range(args.count):
        bundle = out_dir / f"sim_{i:03d}"
        bundle.mkdir(parents=True, exist_ok=True)
        cfg = random_simulation_config(
            sim_index=i,
            motes=args.motes,
            region=tuple(args.region),
            duration=args.duration,
            seed=args.seed + i,
            radius_of_reach=args.radius_of_reach,
            radius_of_inter=args.radius_of_inter,
        )
        out_xml = bundle / "simulation.csc"
        out_dat = bundle / "positions.dat"
        cooja_files.convert_simulation_files(cfg, args.template, str(out_xml), str(out_dat))

        # Copy firmware files (if any) so each container has everything it needs.
        for src in firmware_files:
            dst = bundle / src.name
            dst.write_bytes(src.read_bytes())

        print(f"[prepare] {bundle.name}/ written ({args.motes} motes, seed {args.seed + i})")

    print(f"[prepare] {args.count} bundle(s) ready under {out_dir}/")


if __name__ == "__main__":
    main()
