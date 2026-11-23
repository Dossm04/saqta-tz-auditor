"""Конфигурация LLM: провайдер-независимо через OpenAI-совместимый API."""
from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")  # .env рядом с кодом, независимо от папки запуска
load_dotenv()


def _bool(v: str | None, default: bool) -> bool:
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


PRIVACY_MODES = ("auto", "on", "off")
_LOCAL_NAMES = {"localhost", "host.docker.internal", "gateway.docker.internal", "0.0.0.0"}
_LOCAL_SUFFIXES = (".local", ".localhost", ".internal")


def _privacy_mode(v: str | None) -> str:
    """auto | on | off (а также 1/0/true/false); всё неизвестное → auto."""
    m = (v or "").strip().lower()
    if m in ("1", "true", "yes"):
        return "on"
    if m in ("0", "false", "no"):
        return "off"
    return m if m in PRIVACY_MODES else "auto"


def is_local_url(url: str | None) -> bool:
    """True, если эндпоинт внутри контура: loopback, частная сеть (10/8, 172.16/12, 192.168/16, fc00::/7),
    localhost, *.local, *.localhost, *.internal (host.docker.internal). Пустой base_url = облако OpenAI."""
    if not url:
        return False
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = (parsed.hostname or "").strip().lower().rstrip(".")
    except ValueError:
        return False
    if not host:
        return False
    if host in _LOCAL_NAMES or host.endswith(_LOCAL_SUFFIXES):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


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
    privacy: str = "auto"                   # маскирование ПДн перед LLM: auto (вне контура) | on | off

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def models(self) -> list[str]:
        return list(dict.fromkeys([self.model, *self.fallback_models]))

    @property
    def is_local(self) -> bool:
        """Модель работает внутри контура (Ollama/vLLM на localhost или во внутренней сети)."""
        return is_local_url(self.base_url)

    @property
    def privacy_enabled(self) -> bool:
        """Маскировать ли ИИН/карты/телефоны/e-mail/IBAN перед отправкой в LLM (см. privacy.py)."""
        mode = _privacy_mode(self.privacy)
        if mode == "on":
            return True
        if mode == "off":
            return False
        return not self.is_local

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
            privacy=_privacy_mode(os.getenv("LLM_PRIVACY")),
        )

    def with_overrides(self, **kw) -> "LLMConfig":
        return replace(self, **{k: v for k, v in kw.items() if v not in (None, "")})

    def with_preset(self, name: str) -> "LLMConfig":
        """Применить пресет провайдера из PROVIDER_PRESETS (пустые поля пресета не трогают текущие)."""
        return self.with_overrides(**preset_overrides(name))


# Пресеты провайдеров для выпадающего списка в UI. Поля совпадают с полями LLMConfig
# (кроме служебных: offline, note). Пустая строка / None = «оставить как в .env».
PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "Офлайн-правила": {
        "offline": True,
        "note": "Без LLM: детерминированные правила базы знаний. Текст ТЗ никуда не передаётся.",
    },
    "Локальная модель (Ollama, контур банка)": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",                    # Ollama ключ не проверяет, но клиенту OpenAI он нужен
        "model": "qwen2.5:14b-instruct",
        "fallback_models": (),
        "max_concurrency": 1,
        "json_mode": True,
        "timeout": 600.0,
        "call_budget_s": 900.0,
        "privacy": "auto",                      # localhost → маскирование не требуется
        "note": "Данные не покидают контур банка. Нужен `ollama pull qwen2.5:14b-instruct` и "
                "контекст ≥ 32k (OLLAMA_CONTEXT_LENGTH=32768). См. docs/PRIVACY.md.",
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.5-flash-lite",
        "fallback_models": ("gemini-flash-lite-latest", "gemini-3.1-flash-lite", "gemini-3-flash-preview"),
        "max_concurrency": 3,
        "privacy": "auto",                      # облако → ИИН/карты/телефоны маскируются
        "note": "Бесплатный ключ: aistudio.google.com. Перед отправкой ПДн маскируются.",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "fallback_models": (),
        "max_concurrency": 3,
        "privacy": "auto",
        "note": "Облако OpenAI. Перед отправкой ПДн маскируются.",
    },
    "shai.pro (уточнить на кикоффе)": {
        "base_url": "",                         # endpoint и модель выдадут организаторы
        "model": "",
        "privacy": "auto",
        "note": "Base URL и модель уточнить на кикоффе; введите их вручную.",
    },
}

_PRESET_SERVICE_KEYS = {"offline", "note"}


def preset_overrides(name: str) -> dict[str, Any]:
    """Поля пресета для LLMConfig.with_overrides (без служебных offline/note)."""
    preset = PROVIDER_PRESETS.get(name, {})
    return {k: v for k, v in preset.items() if k not in _PRESET_SERVICE_KEYS}
