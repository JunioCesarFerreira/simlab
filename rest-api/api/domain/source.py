from typing import TypedDict


class SourceFileDto(TypedDict):
    id: str
    file_name: str


class SourceRepositoryDto(TypedDict):
    id: str
    name: str
    description: str
    source_files: list[SourceFileDto]
