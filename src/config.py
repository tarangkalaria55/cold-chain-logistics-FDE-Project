"""Central place that reads the project's .env file.

Importing this module loads ``.env`` from the project root and exposes the
values as a validated, immutable ``settings`` object. Real environment
variables win over the file. Import it before any library that reads
environment variables at import time, such as ``huggingface_hub``.

Secrets are ``SecretStr``: they are masked in logs and ``repr``; call
``.get_secret_value()`` where the real value is needed.
"""
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, NoReturn, Self

from dotenv import load_dotenv
from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "PROJECT_ROOT",
    "AgentLLM",
    "EmbeddingsModel",
    "reveal",
    "settings",
]

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
ENV_FILE: Path = PROJECT_ROOT / ".env"

print(ENV_FILE)

# pydantic-settings reads the file but does not export it, so also load it into
# os.environ for libraries (HF_HUB_OFFLINE, HF_TOKEN, ...) and modules that
# still call os.getenv directly.
load_dotenv(ENV_FILE)


class _ReadOnlyDict(dict[str, Any]):
    """Dict whose mutating methods raise; reads work normally."""

    @staticmethod
    def _blocked() -> NoReturn:
        raise TypeError("settings are read-only")

    def __setitem__(self, *_args: Any, **_kwargs: Any) -> None:
        self._blocked()

    def __delitem__(self, *_args: Any, **_kwargs: Any) -> None:
        self._blocked()

    def __ior__(self, *_args: Any, **_kwargs: Any) -> Self:
        self._blocked()

    def clear(self, *_args: Any, **_kwargs: Any) -> None:
        self._blocked()

    def pop(self, *_args: Any, **_kwargs: Any) -> Any:
        self._blocked()

    def popitem(self, *_args: Any, **_kwargs: Any) -> tuple[str, Any]:
        self._blocked()

    def setdefault(self, *_args: Any, **_kwargs: Any) -> Any:
        self._blocked()

    def update(self, *_args: Any, **_kwargs: Any) -> None:
        self._blocked()


class AgentLLM(StrEnum):
    OLLAMA = "OLLAMA"
    OPENAI = "OPENAI"
    DEEPSEEK = "DEEPSEEK"


class EmbeddingsModel(StrEnum):
    LOCAL = "LOCAL"
    OPENAI = "OPENAI"


class Settings(BaseSettings):
    # Field names match .env keys case-insensitively (Agent_llm -> agent_llm).
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_ignore_empty=True,  # blank values count as unset
        extra="ignore",         # .env also holds keys this class doesn't model
        frozen=True,
        hide_input_in_errors=True,  # never echo secret values in validation errors
    )

    def __init__(self, **values: Any) -> None:
        super().__init__(**values)
        # `frozen=True` blocks `settings.x = ...`; also lock the instance dict so
        # `settings.__dict__["x"] = ...` fails too.
        object.__setattr__(self, "__dict__", _ReadOnlyDict(self.__dict__))

    # Required variables (validation error if missing from .env / environment)
    pinecone_api_key: SecretStr

    # LLM + embeddings routing
    agent_llm: AgentLLM = AgentLLM.OLLAMA
    embeddings_model: EmbeddingsModel = EmbeddingsModel.LOCAL
    local_embedding_model: str = "BAAI/bge-m3"
    openai_model: str = "gpt-4o"
    ollama_model: str = "qwen2.5:3b"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_api_key: SecretStr | None = None  # required only when agent_llm is DEEPSEEK

    # SQL Server
    sql_server_host: str = "localhost"
    sql_server_port: int = 1433
    sql_encrypt: bool = False  # set Encrypt=yes on the SQL connection (TLS) when the server supports it
    sql_agent_user: str = "USR_FDE_RO"
    sql_agent_password: SecretStr | None = None
    sql_admin_user: str | None = None
    sql_admin_password: SecretStr | None = None

    @field_validator("agent_llm", "embeddings_model", mode="before")
    @classmethod
    def _normalise_choice(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("local_embedding_model", "openai_model", "ollama_model", "deepseek_model")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _check_provider_keys(self) -> Self:
        if self.agent_llm is AgentLLM.DEEPSEEK and self.deepseek_api_key is None:
            raise ValueError("DEEPSEEK_API_KEY is required when Agent_llm=DEEPSEEK")
        return self

    @property
    def active_model(self) -> str:
        """Model name for the selected LLM provider."""
        return {
            AgentLLM.OPENAI: self.openai_model,
            AgentLLM.OLLAMA: self.ollama_model,
            AgentLLM.DEEPSEEK: self.deepseek_model,
        }[self.agent_llm]

    def require(self, *names: str) -> None:
        """Raise a clear error if any of the named optional settings is missing."""
        missing = [name for name in names if not getattr(self, name)]
        if missing:
            raise ValueError(f"Missing required setting(s) in {ENV_FILE.name}: {', '.join(missing)}")


def reveal(secret: SecretStr | None) -> str | None:
    """Return the plain value of an optional secret, or None if it isn't set."""
    return secret.get_secret_value() if secret is not None else None


# Cache the instance so .env is parsed once, not on every import.
@lru_cache
def get_settings() -> Settings:
    return Settings()  # values come from the environment/.env, not from arguments


settings: Final[Settings] = get_settings()
