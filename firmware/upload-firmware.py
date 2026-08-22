#!/usr/bin/env python3
# pip install requests
import os

import requests
from pathlib import Path
from typing import List

# ============================================================
# CONFIG
# ============================================================

# SimLab is served only through the nginx reverse proxy over TLS - the REST API
# no longer publishes a host port. Override for a real deployment:
#   SIMLAB_API_BASE=https://simlab.example.org/api/v1
API_BASE = os.getenv("SIMLAB_API_BASE", "https://localhost/api/v1")

# Trust anchor for the proxy's self-signed certificate. It lives in the
# checkout, so verification stays ON by default; set SIMLAB_CA_BUNDLE for a
# different bundle, or SIMLAB_TLS_VERIFY=false to skip verification.
_CA = Path(__file__).resolve().parents[1] / "nginx" / "certs" / "simlab.crt"
VERIFY_TLS = (
    False if os.getenv("SIMLAB_TLS_VERIFY", "true").lower() in ("0", "false", "no", "off")
    else os.getenv("SIMLAB_CA_BUNDLE") or (str(_CA) if _CA.is_file() else True)
)

API_URL = f"{API_BASE}/sources/"
API_KEY = os.getenv("SIMLAB_API_KEY", "simlab-api-key42")  # keep in sync with SIMLAB_API_KEY in .env
FIRMWARE_DIR = Path(".")  # run in firmware/ directory containing rpl-udp-csma/ and rpl-udp-tsch/

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

    headers = {
        "accept": "application/json",
        "X-API-Key": API_KEY,
    }
    
    data = {
        "name": name,
        "description": description,
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            data=data,
            files=files,
            timeout=120,
            verify=VERIFY_TLS,
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