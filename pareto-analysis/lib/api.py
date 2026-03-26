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


def get_generations_from_experiment(
    session: requests.Session,
    api_base: str,
    experiment_id: str,
    label_objectives: list[str]
) -> dict[str, Any]:
    url = f"{api_base}/generations/by-experiment/{experiment_id}"
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    generations: list[dict] = resp.json() or []

    gen_return: dict[int, list[dict]] = {}
    for gen in generations:
        pop: dict[str, Any] = {}
        for ind in gen["population"]:
            ind_id = ind["id"]
            pop[ind_id] = {
                "id": ind_id,
                "objectives": {k: v for k, v in zip(label_objectives, ind["objectives"])}
            }
        gen_return[gen["index"]] = list(pop.values())

    return gen_return


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

