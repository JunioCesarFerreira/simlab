import os
from dataclasses import dataclass


def env_to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class MongoServiceSettings:
    mongo_uri: str
    db_name: str

    @classmethod
    def from_env(cls) -> "MongoServiceSettings":
        return cls(
            mongo_uri=os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0"),
            db_name=os.getenv("DB_NAME", "simlab"),
        )


@dataclass(frozen=True)
class ApiAuthSettings:
    api_key: str

    @classmethod
    def from_env(cls) -> "ApiAuthSettings":
        return cls(api_key=os.getenv("SIMLAB_API_KEY", "api-password"))


@dataclass(frozen=True)
class CoojaTemplateSettings:
    template_xml: str
    is_docker: bool

    @classmethod
    def from_env(cls) -> "CoojaTemplateSettings":
        return cls(
            template_xml=os.getenv("TEMPLATE_XML", "./simulation_template.xml"),
            is_docker=env_to_bool(os.getenv("IS_DOCKER"), False),
        )
