#!/usr/bin/env python3
# pip install requests
import os
import sys

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
FIRMWARE_DIR = Path(".")  # run in the firmware/ directory, which holds the rpl-udp-* directories

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

# The repository name is the directory name. v1 is what every experiment run so
# far used and must not be edited (its digests are pinned by a test); v2 changes
# how the metrics are obtained. See docs/markdown/FIRMWARE_METRICS.md.
# rpl-udp-tsch-6o is an older variant that no experiment used and is not uploaded.
REPOSITORIES = {
    "rpl-udp-csma": "RPL UDP firmware using CSMA MAC layer (v1)",
    "rpl-udp-tsch": "RPL UDP firmware using TSCH MAC layer (v1)",
    "rpl-udp-csma-v2": "RPL UDP firmware using CSMA MAC layer (v2: no probes, one-way latency, cumulative counters)",
    "rpl-udp-tsch-v2": "RPL UDP firmware using TSCH MAC layer (v2: no probes, one-way latency, cumulative counters, shared cell every 3 slots)",
}


def main():
    """Upload the repositories named on the command line, or all of them."""
    names = sys.argv[1:] or list(REPOSITORIES)

    unknown = [n for n in names if n not in REPOSITORIES]
    if unknown:
        raise SystemExit(f"Unknown repository: {', '.join(unknown)}. Known: {', '.join(REPOSITORIES)}")

    # Check every directory before uploading any, so a typo does not leave half the set uploaded.
    for name in names:
        if not (FIRMWARE_DIR / name).exists():
            raise RuntimeError(f"Directory {name} not found")

    for name in names:
        create_repository(
            name=name,
            description=REPOSITORIES[name],
            directory=FIRMWARE_DIR / name,
        )


if __name__ == "__main__":
    main()