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


def get_generations_from_experiment(
    session: requests.Session,
    api_base: str,
    experiment_id: str,
    label_objectives: list[str]
) -> dict[int, list[dict]]:
    url = f"{api_base}/generations/by-experiment/{experiment_id}"
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    generations: list[dict] = resp.json() or []

    gen_return: dict[int, list[dict]] = {}
    for gen in generations:
        valid: list[dict] = []
        for ind in gen["population"]:
            raw_objs: list[float] = ind.get("objectives", [])
            # Individuals that never got a full objective vector (still running,
            # or a partial write) are dropped, matching the hv-gd endpoint's
            # `len(raw) < n_obj` guard — without it the zip below would silently
            # build a short objectives dict and KeyError later.
            if len(raw_objs) < len(label_objectives):
                continue
            # Penalty, placeholder or non-finite: decided by the evidence stored
            # with the vector; by magnitude only for legacy individuals.
            if is_excluded(raw_objs, ind.get("completeness")):
                continue
            valid.append({
                "id": ind["id"],
                "objectives": {k: v for k, v in zip(label_objectives, raw_objs)},
            })
        gen_return[gen["index"]] = valid

    return gen_return


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
