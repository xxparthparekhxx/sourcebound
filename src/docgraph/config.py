"""Configuration: one TOML file plus environment variables.

API keys are never stored in the config. An endpoint names the environment
variable that holds its key (``api_key_env``); the key is read from the
environment at request time.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ConfigError(ValueError):
    """The configuration is invalid or incomplete."""


_SECRET_KEYS = {"api_key", "apikey", "key", "token", "secret", "password"}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _no_literal_secrets(cls, data: Any) -> Any:
        if isinstance(data, dict):
            bad = sorted(k for k in data if k.lower() in _SECRET_KEYS)
            if bad:
                raise ValueError(
                    f"{bad[0]!r} must not be set in the config file: keys come from environment "
                    "variables only. Use api_key_env = \"NAME_OF_ENV_VAR\" instead."
                )
        return data


class EndpointConfig(_Model):
    """One OpenAI-compatible chat endpoint."""

    name: str | None = None
    base_url: str
    model: str
    api_key_env: str | None = None
    # Context window of the served model, in tokens. When set, a request whose estimated size
    # (prompt + max_tokens) exceeds it is never sent: the endpoint is skipped, and the caller gets
    # PromptTooLong if no endpoint fits. This matters for Ollama, which silently truncates prompts
    # to its context length (4096 by default).
    context_tokens: int | None = None
    # Merged into every chat request body for this endpoint, e.g.
    # {temperature = 0.7, top_p = 0.8, top_k = 20, reasoning_effort = "none"}.
    # These win over the defaults a caller passes, so they are the place to put a
    # model's recommended sampling settings and to switch thinking off.
    options: dict[str, Any] = Field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.name or self.base_url

    def api_key(self, env: Mapping[str, str] | None = None) -> str | None:
        if not self.api_key_env:
            return None
        return (env if env is not None else os.environ).get(self.api_key_env) or None

    def available(self, env: Mapping[str, str] | None = None) -> bool:
        """An endpoint that names a key variable is skipped while that variable is unset."""
        return self.api_key_env is None or self.api_key(env) is not None


class ChatConfig(_Model):
    endpoints: list[EndpointConfig] = Field(
        default_factory=lambda: [
            EndpointConfig(
                name="ollama",
                base_url="http://localhost:11434/v1",
                model="bonsai-2",
                context_tokens=4096,
                options={"reasoning_effort": "none"},
            )
        ]
    )
    timeout: float = 120.0
    max_tokens: int = 1024
    # Passes over the endpoint list before giving up (waits between passes; useful with rate limits).
    passes: int = 1
    backoff: float = 2.0


class EmbeddingsConfig(_Model):
    provider: Literal["local", "openai"] = "local"
    model: str = "BAAI/bge-small-en-v1.5"
    # provider = "openai": any OpenAI-compatible /v1/embeddings endpoint
    base_url: str | None = None
    api_key_env: str | None = None
    query_prefix: str = ""
    document_prefix: str = ""
    batch_size: int = 32
    timeout: float = 120.0

    @model_validator(mode="after")
    def _check(self) -> "EmbeddingsConfig":
        if self.provider == "openai" and not self.base_url:
            raise ValueError("embeddings.base_url is required when embeddings.provider = \"openai\"")
        return self


class OcrConfig(_Model):
    enabled: bool = False
    # Name of one of the chat endpoints; default is the first available one.
    endpoint: str | None = None
    # Vision-capable model on that endpoint; default is the endpoint's own model.
    model: str | None = None
    dpi: int = 150
    min_chars: int = 20  # a page with fewer extractable characters counts as "no text layer"


class GraphConfig(_Model):
    enabled: bool = False  # extract entities and relations at ingest time (one LLM call per chunk)
    workers: int = 2
    max_entities: int = 12
    max_relations: int = 12


class IngestConfig(_Model):
    chunk_size: int = 1200  # characters
    chunk_overlap: int = 200  # characters
    context_header: bool = True  # prefix "title > heading" to the text that is embedded
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)

    @model_validator(mode="after")
    def _check(self) -> "IngestConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("ingest.chunk_overlap must be smaller than ingest.chunk_size")
        return self


class RetrievalConfig(_Model):
    """Every stage can be switched off independently (the eval does exactly that)."""

    vector: bool = True
    bm25: bool = True
    mmr: bool = True
    graph: bool = False
    rerank: bool = False
    top_k: int = 5  # passages handed to the answerer
    candidates: int = 20  # per-source depth fed to fusion
    vector_fetch: int = 40  # vector hits fetched before MMR picks `candidates`
    mmr_lambda: float = 0.7
    rrf_k: int = 60
    graph_max_chunks: int = 20
    rerank_candidates: int = 20
    # "Nothing relevant" gate (see retrieval.is_relevant)
    min_vector_score: float = 0.55
    min_term_overlap: float = 0.5


class RerankConfig(_Model):
    model: str = "xxparthparekhxx/openjev-flash"
    device: str = "cpu"
    hypothesis: str = 'The passage answers the question "{question}": it states the answer explicitly.'
    batch_size: int = 16
    max_length: int = 1024
    # Faithfulness: a sentence counts as supported when P(entailment) >= this.
    entailment_threshold: float = 0.5


class AnswerConfig(_Model):
    mode: Literal["pipeline", "agent"] = "pipeline"
    max_steps: int = 4
    max_passage_chars: int = 1500  # per passage in the prompt
    no_answer_message: str = "I could not find anything relevant to that question in the indexed documents."


class Config(_Model):
    data_dir: Path = Path(".docgraph")
    chat: ChatConfig = Field(default_factory=ChatConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    answer: AnswerConfig = Field(default_factory=AnswerConfig)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "docgraph.db"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"


def load_dotenv(path: str | Path = ".env", env: dict[str, str] | None = None) -> bool:
    """Minimal .env reader: KEY=VALUE lines, never overrides a variable that is already set."""
    target = os.environ if env is None else env
    p = Path(path)
    if not p.is_file():
        return False
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in target:
            target[key] = value
    return True


def find_config_file(explicit: str | Path | None, env: Mapping[str, str]) -> Path | None:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        return p
    if env.get("DOCGRAPH_CONFIG"):
        p = Path(env["DOCGRAPH_CONFIG"])
        if not p.is_file():
            raise ConfigError(f"DOCGRAPH_CONFIG points to a missing file: {p}")
        return p
    for candidate in (Path("docgraph.toml"), Path.home() / ".config" / "docgraph" / "config.toml"):
        if candidate.is_file():
            return candidate
    return None


def _apply_env(raw: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """Environment overrides. Only simple, documented variables; everything else lives in the file."""
    if env.get("DOCGRAPH_DATA_DIR"):
        raw["data_dir"] = env["DOCGRAPH_DATA_DIR"]

    # One-endpoint shortcut: DOCGRAPH_BASE_URL (+ DOCGRAPH_MODEL, DOCGRAPH_API_KEY_ENV)
    # replaces the endpoint list. Handy for "just point it at my server".
    if env.get("DOCGRAPH_BASE_URL"):
        ep: dict[str, Any] = {"base_url": env["DOCGRAPH_BASE_URL"], "name": "env"}
        ep["model"] = env.get("DOCGRAPH_MODEL") or "default"
        if env.get("DOCGRAPH_API_KEY_ENV"):
            ep["api_key_env"] = env["DOCGRAPH_API_KEY_ENV"]
        raw.setdefault("chat", {})["endpoints"] = [ep]
    elif env.get("DOCGRAPH_MODEL"):
        eps = raw.setdefault("chat", {}).get("endpoints")
        if eps:
            eps[0]["model"] = env["DOCGRAPH_MODEL"]

    emb = raw.setdefault("embeddings", {})
    for var, field in (
        ("DOCGRAPH_EMBED_PROVIDER", "provider"),
        ("DOCGRAPH_EMBED_MODEL", "model"),
        ("DOCGRAPH_EMBED_BASE_URL", "base_url"),
        ("DOCGRAPH_EMBED_API_KEY_ENV", "api_key_env"),
    ):
        if env.get(var):
            emb[field] = env[var]
    if env.get("DOCGRAPH_EMBED_BASE_URL") and "provider" not in emb:
        emb["provider"] = "openai"
    return raw


def load_config(path: str | Path | None = None, env: Mapping[str, str] | None = None) -> Config:
    env = os.environ if env is None else env
    raw: dict[str, Any] = {}
    file = find_config_file(path, env)
    if file is not None:
        try:
            raw = tomllib.loads(file.read_text())
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"{file}: invalid TOML: {e}") from e
    # `[[chat.endpoints]]` is the documented spelling.
    raw = _apply_env(raw, env)
    try:
        cfg = Config.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"invalid configuration{f' in {file}' if file else ''}:\n{_fmt(e)}") from e
    return cfg


def _fmt(e: ValidationError) -> str:
    lines = []
    for err in e.errors():
        loc = ".".join(str(p) for p in err["loc"])
        lines.append(f"  {loc}: {err['msg']}")
    return "\n".join(lines)
