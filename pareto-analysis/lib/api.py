import os
import requests
from pathlib import Path
from typing import Any

# Individuals whose ANY objective (absolute value) exceeds this threshold
# are considered penalized and excluded from all analyses.
PENALTY_THRESHOLD = 1e8

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


def _is_penalized(raw_objectives: list[float]) -> bool:
    return any(abs(v) >= PENALTY_THRESHOLD for v in raw_objectives)


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
            if _is_penalized(raw_objs):
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
    return resp.json().get("pareto_front") or []


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
