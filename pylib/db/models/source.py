from typing import TypedDict


class SourceFile(TypedDict):
    id: str
    file_name: str


class SourceRepository(TypedDict):
    id: str
    name: str
    description: str
    source_files: list[SourceFile]
