"""Minimal Prometheus HTTP API client (stdlib only)."""
import json
import logging
import urllib.parse
import urllib.request
from datetime import datetime

logger = logging.getLogger(__name__)


class PrometheusError(RuntimeError):
    """Raised when Prometheus is unreachable or returns an error payload."""


class PrometheusClient:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self, timeout: float = 5.0) -> bool:
        """True when the Prometheus health endpoint answers.

        Cheap pre-flight used to skip collection with a single warning line
        (instead of a traceback) when the server is unreachable — e.g. a local
        run without the monitoring stack.
        """
        url = f"{self.base_url}/-/healthy"
        try:
            with urllib.request.urlopen(url, timeout=min(self.timeout, timeout)) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False

    def query_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str = "15s",
    ) -> list[dict]:
        """Run ``/api/v1/query_range`` and return the raw ``result`` list.

        Each entry is a Prometheus *matrix* series:
        ``{"metric": {<labels>}, "values": [[<epoch>, "<value>"], ...]}``.
        """
        params = urllib.parse.urlencode({
            "query": query,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": step,
        })
        url = f"{self.base_url}/api/v1/query_range?{params}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise PrometheusError(f"query_range request failed: {e}") from e

        if payload.get("status") != "success":
            raise PrometheusError(
                f"query_range returned status={payload.get('status')!r}: "
                f"{payload.get('error', 'unknown error')}"
            )
        return payload.get("data", {}).get("result", [])
