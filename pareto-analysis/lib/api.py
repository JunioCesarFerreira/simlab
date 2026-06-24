import requests
from pathlib import Path
from typing import Any

# Individuals whose ANY objective (absolute value) exceeds this threshold
# are considered penalized and excluded from all analyses.
PENALTY_THRESHOLD = 1e8


def build_session(api_key: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "accept": "application/json",
        "X-API-Key": api_key,
    })
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
