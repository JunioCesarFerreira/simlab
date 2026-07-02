from typing import TypedDict, Optional


class SshWorker(TypedDict):
    hostname: str
    port: int
    username: str
    password: str
    enabled: bool
    label: Optional[str]
