import os
from dataclasses import dataclass
from typing import Literal, Optional

from openai import AsyncOpenAI


Provider = Literal["openai", "alice"]


@dataclass(frozen=True)
class ProviderConfig:
    provider: Provider
    api_key: str
    chat_base_url: Optional[str]
    embed_base_url: Optional[str]
    chat_model: str
    embed_model: str


def _env(name: str) -> str:
    v = os.getenv(name)
    return "" if v is None else str(v).strip()


def detect_provider() -> ProviderConfig:
    """Detect whether to use OpenAI direct or ALICE gateway.

    Selection rule (simple + matches your .env comment workflow):
    - If OPENAI_API_KEY is set -> use OpenAI (no base_url required)
    - Else if ALICE_OPENAI_KEY is set -> use ALICE (requires base_url)
    - Else -> raise

    Model override env vars (optional):
    - CHAT_MODEL (global override)
    - EMBED_MODEL (global override)
    - OPENAI_CHAT_MODEL / OPENAI_EMBED_MODEL
    - ALICE_CHAT_MODEL / ALICE_EMBED_MODEL
    """

    openai_key = _env("OPENAI_API_KEY")
    alice_key = _env("ALICE_OPENAI_KEY")

    if openai_key:
        chat_model = _env("OPENAI_CHAT_MODEL") or _env("CHAT_MODEL") or "gpt-4o-mini"
        embed_model = _env("OPENAI_EMBED_MODEL") or _env("EMBED_MODEL") or "text-embedding-3-small"

        # OpenAI direct API expects model IDs like 'gpt-4o-mini' (no 'openai/' prefix).
        if chat_model.startswith("openai/"):
            chat_model = chat_model[len("openai/"):]

        return ProviderConfig(
            provider="openai",
            api_key=openai_key,
            chat_base_url=None,
            embed_base_url=None,
            chat_model=chat_model,
            embed_model=embed_model,
        )

    if alice_key:
        chat_base_url = _env("ALICE_CHAT_URL")
        embed_base_url = _env("ALICE_EMBED_URL")
        chat_model = _env("ALICE_CHAT_MODEL") or _env("CHAT_MODEL") or "openai/gpt-4o-mini"
        embed_model = _env("ALICE_EMBED_MODEL") or _env("EMBED_MODEL") or "text-embedding-3-small"
        return ProviderConfig(
            provider="alice",
            api_key=alice_key,
            chat_base_url=chat_base_url or None,
            embed_base_url=embed_base_url or None,
            chat_model=chat_model,
            embed_model=embed_model,
        )

    raise RuntimeError(
        "No API key found. Set either OPENAI_API_KEY (OpenAI) or ALICE_OPENAI_KEY (ALICE) in your environment/.env."
    )


def create_async_chat_client(cfg: Optional[ProviderConfig] = None) -> AsyncOpenAI:
    cfg = cfg or detect_provider()
    if cfg.provider == "openai":
        return AsyncOpenAI(api_key=cfg.api_key)

    if not cfg.chat_base_url:
        raise RuntimeError(
            "ALICE provider selected (ALICE_OPENAI_KEY is set) but ALICE_CHAT_URL is missing."
        )
    return AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.chat_base_url)


def create_async_embed_client(cfg: Optional[ProviderConfig] = None) -> AsyncOpenAI:
    cfg = cfg or detect_provider()
    if cfg.provider == "openai":
        return AsyncOpenAI(api_key=cfg.api_key)

    if not cfg.embed_base_url:
        raise RuntimeError(
            "ALICE provider selected (ALICE_OPENAI_KEY is set) but ALICE_EMBED_URL is missing."
        )
    return AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.embed_base_url)
