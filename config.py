"""Конфигурация LLM: провайдер-независимо через OpenAI-совместимый API."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")  # .env рядом с кодом, независимо от папки запуска
load_dotenv()


def _bool(v: str | None, default: bool) -> bool:
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class LLMConfig:
    base_url: str | None = None
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    json_mode: bool = True          # response_format=json_object (выключить, если провайдер не умеет)
    timeout: float = 180.0
    max_doc_chars: int = 80_000     # защита от слишком длинных ТЗ
    max_retries: int = 2
    fallback_models: tuple[str, ...] = ()   # резервные модели при 429/503/404/тайм-ауте
    max_concurrency: int = 3                # одновременных запросов (лимиты бесплатных тарифов)
    call_budget_s: float = 150.0            # сколько секунд агент ждёт провайдера, прежде чем уйти на правила
    cache: bool = True                      # кэшировать ответы LLM на диск (.cache/llm)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def models(self) -> list[str]:
        return list(dict.fromkeys([self.model, *self.fallback_models]))

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            base_url=os.getenv("LLM_BASE_URL") or None,
            api_key=os.getenv("LLM_API_KEY") or None,
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            json_mode=_bool(os.getenv("LLM_JSON_MODE"), True),
            timeout=float(os.getenv("LLM_TIMEOUT", "180")),
            max_doc_chars=int(os.getenv("LLM_MAX_DOC_CHARS", "80000")),
            fallback_models=tuple(m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "").split(",") if m.strip()),
            max_concurrency=int(os.getenv("LLM_MAX_CONCURRENCY", "3")),
            call_budget_s=float(os.getenv("LLM_CALL_BUDGET", "150")),
            cache=_bool(os.getenv("LLM_CACHE"), True),
        )

    def with_overrides(self, **kw) -> "LLMConfig":
        return replace(self, **{k: v for k, v in kw.items() if v not in (None, "")})
