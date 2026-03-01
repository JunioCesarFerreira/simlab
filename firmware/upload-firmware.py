#!/usr/bin/env python3

import requests
from pathlib import Path
from typing import List

# ============================================================
# CONFIG
# ============================================================

API_URL = "http://localhost:8198/api/v1/source-repository/"
FIRMWARE_DIR = Path(".")  # executar dentro de firmware/


# ============================================================
# Helpers
# ============================================================

def collect_files(directory: Path) -> List[tuple]:
    """
    Collect all files inside directory for multipart upload.
    Returns list compatible with requests 'files=' argument.
    """
    files = []

    for f in directory.iterdir():
        if f.is_file():
            files.append(
                (
                    "files",
                    (
                        f.name,
                        open(f, "rb"),
                        "application/octet-stream",
                    ),
                )
            )

    return files


def create_repository(name: str, description: str, directory: Path):
    """
    Sends POST request creating repository from directory files.
    """

    print(f"\nUploading repository: {name}")

    files = collect_files(directory)

    data = {
        "name": name,
        "description": description,
    }

    try:
        response = requests.post(
            API_URL,
            data=data,
            files=files,
            timeout=120,
        )

        response.raise_for_status()

        repo_id = response.text.strip('"')
        print(f"✅ Created repository {name}")
        print(f"   repository_id = {repo_id}")

    finally:
        # important: close file handlers
        for _, file_tuple in files:
            file_tuple[1].close()


# ============================================================
# MAIN
# ============================================================

def main():

    csma_dir = FIRMWARE_DIR / "rpl-udp-csma"
    tsch_dir = FIRMWARE_DIR / "rpl-udp-tsch"

    if not csma_dir.exists():
        raise RuntimeError("Directory rpl-udp-csma not found")

    if not tsch_dir.exists():
        raise RuntimeError("Directory rpl-udp-tsch not found")

    # --------------------------------------------------------
    # POST 1 — CSMA
    # --------------------------------------------------------
    create_repository(
        name="rpl-udp-csma",
        description="RPL UDP firmware using CSMA MAC layer",
        directory=csma_dir,
    )

    # --------------------------------------------------------
    # POST 2 — TSCH
    # --------------------------------------------------------
    create_repository(
        name="rpl-udp-tsch",
        description="RPL UDP firmware using TSCH MAC layer",
        directory=tsch_dir,
    )


if __name__ == "__main__":
    main()