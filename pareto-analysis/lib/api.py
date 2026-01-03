import requests
from pathlib import Path
from typing import Any

def build_session(api_key: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "accept": "application/json",
        "X-API-Key": api_key,
    })
    return s


def get_pareto_per_generation_api(
    session: requests.Session,
    api_base: str,
    experiment_id: str,
    to_minimization: bool = False
) -> dict[int, list[dict[str, Any]]]:
    url = f"{api_base}/analysis/experiments/{experiment_id}/paretos"
    if to_minimization:
        url += "_to_min"
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # JSON keys come as strings; normalize to int
    return {int(k): v for k, v in data.items()}


def upload_analysis_file_api(
    session: requests.Session,
    api_base: str,
    experiment_id: str,
    path: Path,
    name: str,
    description: str
) -> None:
    """
    Upload an analysis file and attach it to an experiment.
    Matches the API contract used by curl.
    """
    url = f"{api_base}/experiments/{experiment_id}/analysis-file"

    with open(path, "rb") as f:
        files = {
            "file": (path.name, f, "image/png")
        }
        data = {
            "name": name,
            "description": description
        }

        resp = session.patch(
            url,
            files=files,
            data=data,
            timeout=120
        )

    resp.raise_for_status()

