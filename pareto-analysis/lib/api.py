import os
import requests
from pathlib import Path
from typing import Any

from lib.metrics import PENALTY_THRESHOLD, is_excluded, is_valid_evaluation  # noqa: F401

# The REST API no longer publishes a host port: everything goes through the
# nginx reverse proxy over TLS. Override per environment with SIMLAB_API_BASE.
DEFAULT_API_BASE = os.getenv("SIMLAB_API_BASE", "https://localhost/api/v1")

# Trust anchor for the proxy's self-signed certificate. These scripts run from
# a checkout, so the certificate the proxy-certs container wrote is sitting
# right there - pointing requests at it keeps verification ON instead of
# turning it off. Irrelevant once a CA-issued certificate is installed.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CA_BUNDLE = _REPO_ROOT / "nginx" / "certs" / "simlab.crt"

_FALSEY = {"0", "false", "no", "off"}


def resolve_tls_verify() -> bool | str:
    """Value for the ``verify=`` argument of requests.

    Resolution order:
      1. ``SIMLAB_TLS_VERIFY`` falsey -> no verification at all (last resort).
      2. ``SIMLAB_CA_BUNDLE`` -> that file.
      3. ``nginx/certs/simlab.crt`` when present -> the self-signed proxy cert.
      4. the system trust store, for a CA-issued certificate or plain HTTP.
    """
    if os.getenv("SIMLAB_TLS_VERIFY", "true").strip().lower() in _FALSEY:
        return False

    bundle = os.getenv("SIMLAB_CA_BUNDLE", "").strip()
    if bundle:
        return bundle

    if _DEFAULT_CA_BUNDLE.is_file():
        return str(_DEFAULT_CA_BUNDLE)

    return True


def build_session(api_key: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "accept": "application/json",
        "X-API-Key": api_key,
    })
    s.verify = resolve_tls_verify()
    return s


def get_experiment(session: requests.Session, api_base: str, experiment_id: str) -> dict:
    url = f"{api_base}/experiments/{experiment_id}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json() or {}


def declared_objectives(experiment: dict) -> list[str]:
    return [str(o.get("metric_name")) for o in
            ((experiment.get("parameters") or {}).get("objectives") or [])
            if o.get("metric_name")]


#: Benchmarks with an analytical front here (lib/true_fronts.py).
KNOWN_BENCHES = ("DTLZ2", "ZDT1", "SCH1")


def synthetic_bench(experiment: dict, objectives: list[str], minimize: list[bool]) -> str | None:
    """The benchmark whose analytical front /hv-gd measures this request against.

    The same conditions as the endpoint: a synthetic experiment, every
    objective minimised, and the whole set requested, in any order. The front
    of a projection onto fewer objectives is not the front of the smaller
    benchmark, so a subset keeps the empirical reference.
    """
    syn = (((experiment.get("parameters") or {}).get("simulation") or {}).get("synthetic") or {})
    bench = str(syn.get("bench") or "").upper()
    if not (syn.get("enabled") and bench in KNOWN_BENCHES and all(minimize)):
        return None
    declared = declared_objectives(experiment)
    if declared and sorted(objectives) != sorted(declared):
        return None
    return bench


def canonical_order(objectives: list[str], minimize: list[bool], experiment: dict):
    """A permutation of the declared objectives, put back in the declared order.

    HV, GD, IGD and IGD+ do not depend on the order of the axes, and the
    analytical front has one: measured in the declared order, a permuted
    request compares each objective with its own column of the front.
    """
    declared = declared_objectives(experiment)
    if not declared or sorted(objectives) != sorted(declared) or objectives == declared:
        return list(objectives), list(minimize)
    goal = dict(zip(objectives, minimize))
    return declared, [goal[o] for o in declared]


def objective_columns(requested: list[str], experiment: dict) -> list[int]:
    """Where each requested objective sits in the stored vectors.

    Individuals store their objectives as a positional list, in the order the
    experiment declares them. Pairing them with ``--objectives`` in the order
    given put the wrong numbers under each name whenever that order differed
    or was a subset. Experiments that declare no objectives leave nothing to
    resolve against, and positional order is all there is.
    """
    declared = declared_objectives(experiment)
    if not declared:
        return list(range(len(requested)))
    unknown = [o for o in requested if o not in declared]
    if unknown:
        raise SystemExit(
            f"[ERROR] Unknown objective(s) {unknown}; this experiment declares {declared}.")
    return [declared.index(o) for o in requested]


def load_generations(
    session: requests.Session,
    api_base: str,
    experiment_id: str,
    label_objectives: list[str],
    columns: list[int] | None = None,
) -> tuple[dict[int, list[dict]], dict[int, list[str]]]:
    """The usable individuals of each generation, and each generation's
    survivors (chromosome hashes) where it recorded them."""
    columns = columns if columns is not None else list(range(len(label_objectives)))
    url = f"{api_base}/generations/by-experiment/{experiment_id}"
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    generations: list[dict] = resp.json() or []

    gen_return: dict[int, list[dict]] = {}
    survivors: dict[int, list[str]] = {}
    for gen in generations:
        valid: list[dict] = []
        for ind in gen["population"]:
            raw_objs: list[float] = ind.get("objectives", [])
            # Individuals that never got a full objective vector (still running,
            # or a partial write) are dropped, matching the hv-gd endpoint's
            # guard — without it the lookup below would fail later.
            if len(raw_objs) <= max(columns, default=-1):
                continue
            # Penalty, placeholder or non-finite: decided by the evidence stored
            # with the vector; by magnitude only for legacy individuals.
            if is_excluded(raw_objs, ind.get("completeness")):
                continue
            valid.append({
                "id": ind["id"],
                "individual_id": ind.get("individual_id"),
                "objectives": {k: raw_objs[c] for k, c in zip(label_objectives, columns)},
            })
        gen_return[gen["index"]] = valid
        if gen.get("survivors") is not None:
            survivors[gen["index"]] = list(gen["survivors"])

    return gen_return, survivors


def get_generations_from_experiment(
    session: requests.Session,
    api_base: str,
    experiment_id: str,
    label_objectives: list[str],
    columns: list[int] | None = None,
) -> dict[int, list[dict]]:
    return load_generations(session, api_base, experiment_id, label_objectives, columns)[0]


def get_experiment_pareto_front(
    session: requests.Session,
    api_base: str,
    experiment_id: str,
) -> list[dict]:
    """
    Return the Pareto front stored on the experiment document.

    Each item: {"objectives": {metric_name: float, ...}, "chromosome": {...}}
    This is the authoritative Pareto front computed by the optimization engine
    and is exactly what the GUI displays.
    """
    url = f"{api_base}/experiments/{experiment_id}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    items = resp.json().get("pareto_front") or []
    # Items the engine marked as a penalty or a placeholder are not part of a
    # result. Items written before the mark existed carry no `evaluation`, and
    # the reference-front sanitiser applies the legacy magnitude rule to them.
    return [item for item in items if is_valid_evaluation(item.get("evaluation"))]


def upload_analysis_file_api(
    session: requests.Session,
    api_base: str,
    experiment_id: str,
    path: Path,
    name: str,
    description: str
) -> None:
    url = f"{api_base}/experiments/{experiment_id}/analysis-file"

    with open(path, "rb") as f:
        files = {"file": (path.name, f, "image/png")}
        data = {"name": name, "description": description}
        resp = session.patch(url, files=files, data=data, timeout=120)

    resp.raise_for_status()
