"""Pydantic-схемы: находки, вопросы, модель документа, итоговый отчёт.

Все агенты обмениваются строго типизированными объектами — это делает
отчёт детерминированным и позволяет проверять вывод LLM.
"""
from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── нормализация списков из ответа LLM ─────────────────────────────────────────
# Модель нередко отдаёт скаляр вместо списка («lines": 12, "evidence": "цитата") — это не должно
# ронять весь ответ агента или превращать «123» в строки [1, 2, 3].

def _as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return list(v)
    if isinstance(v, dict):
        return list(v.values())
    return [v]


def to_int_list(v: Any) -> list[int]:
    """Номера строк: 12 → [12]; "L12, L14" → [12, 14]; ["L3", 5.0, "x"] → [3, 5]."""
    if isinstance(v, str):
        return [int(x) for x in re.findall(r"\d+", v)]
    out: list[int] = []
    for x in _as_list(v):
        if isinstance(x, bool):
            continue
        if isinstance(x, int) or (isinstance(x, float) and math.isfinite(x)):
            out.append(int(x))
        elif isinstance(x, str):
            out.extend(int(d) for d in re.findall(r"\d+", x))
    return out


def to_str_list(v: Any) -> list[str]:
    """Список строк: "цитата" → ["цитата"]; 7 → ["7"]; пустые элементы отбрасываются."""
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    return [str(x).strip() for x in _as_list(v) if x is not None and str(x).strip()]


class Severity(str, Enum):
    critical = "critical"
    major = "major"
    minor = "minor"

    @property
    def label(self) -> str:
        return {"critical": "Критично", "major": "Важно", "minor": "Мелочь"}[self.value]

    @property
    def icon(self) -> str:
        return {"critical": "🔴", "major": "🟠", "minor": "🟡"}[self.value]

    @property
    def rank(self) -> int:
        return {"critical": 0, "major": 1, "minor": 2}[self.value]


_SEVERITY_ALIASES = {
    "critical": "critical", "критично": "critical", "критичный": "critical", "критическая": "critical",
    "blocker": "critical", "high": "critical", "блокер": "critical",
    "major": "major", "важно": "major", "важный": "major", "medium": "major", "средняя": "major",
    "minor": "minor", "мелочь": "minor", "low": "minor", "низкая": "minor", "info": "minor",
}


class Category(str, Enum):
    completeness = "completeness"     # нет раздела / требования
    ambiguity = "ambiguity"           # размытые формулировки
    contradiction = "contradiction"   # противоречия, дубли
    testability = "testability"       # нельзя проверить / нет критериев приёмки
    security = "security"             # ИБ
    compliance = "compliance"         # регулятор, законы РК, AML/KYC, PCI DSS
    fintech = "fintech"               # финтех-логика: лимиты, идемпотентность, сверка

    @property
    def label(self) -> str:
        return {
            "completeness": "Полнота",
            "ambiguity": "Неоднозначность",
            "contradiction": "Противоречие",
            "testability": "Тестируемость",
            "security": "Безопасность",
            "compliance": "Регулятор / комплаенс",
            "fintech": "Финтех-логика",
        }[self.value]


_CATEGORY_ALIASES = {
    "duplicate": "contradiction", "consistency": "contradiction", "conflict": "contradiction",
    "clarity": "ambiguity", "vague": "ambiguity",
    "regulatory": "compliance", "regulation": "compliance", "aml": "compliance", "kyc": "compliance",
    "privacy": "compliance", "infosec": "security", "appsec": "security",
    "transaction": "fintech", "payments": "fintech", "business": "fintech",
    "missing": "completeness", "acceptance": "testability", "nfr": "testability",
}


class Finding(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = ""
    agent: str = ""
    category: Category = Category.completeness
    severity: Severity = Severity.major
    title: str
    description: str = ""
    evidence: list[str] = Field(default_factory=list)   # дословные цитаты из ТЗ
    lines: list[int] = Field(default_factory=list)       # номера строк L#
    regulation: str | None = None                        # ссылка на норму / стандарт
    recommendation: str = ""
    suggested_text: str | None = None                    # предлагаемая формулировка требования
    question: str | None = None                          # черновик уточняющего вопроса
    control_id: str | None = None                        # ID контроля из базы знаний
    confidence: float = 0.8
    sources: list[str] = Field(default_factory=list)     # какие агенты нашли
    verified: bool | None = None
    verification_note: str | None = None
    revision: int = 0                                    # 1 — исправлена агентом по замечанию Verifier

    @field_validator("severity", mode="before")
    @classmethod
    def _norm_sev(cls, v: Any) -> Any:
        if isinstance(v, str):
            return _SEVERITY_ALIASES.get(v.strip().lower(), v.strip().lower())
        return v

    @field_validator("category", mode="before")
    @classmethod
    def _norm_cat(cls, v: Any) -> Any:
        if isinstance(v, str):
            key = v.strip().lower()
            if key in Category._value2member_map_:
                return key
            return _CATEGORY_ALIASES.get(key, "completeness")
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def _norm_conf(cls, v: Any) -> Any:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.7
        return max(0.0, min(1.0, v if v <= 1 else v / 100))

    @field_validator("lines", mode="before")
    @classmethod
    def _norm_lines(cls, v: Any) -> Any:
        return to_int_list(v)

    @field_validator("evidence", mode="before")
    @classmethod
    def _norm_evidence(cls, v: Any) -> Any:
        return to_str_list(v)


class Priority(str, Enum):
    P0 = "P0"  # блокирует разработку
    P1 = "P1"  # нужно до старта спринта
    P2 = "P2"  # можно уточнить по ходу

    @property
    def label(self) -> str:
        return {"P0": "P0 — блокер", "P1": "P1 — до старта", "P2": "P2 — по ходу"}[self.value]


class Question(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = ""
    question: str
    why: str = ""                                        # зачем спрашиваем / риск
    priority: Priority = Priority.P1
    addressee: str = "Бизнес-заказчик"                   # кому адресован
    related_findings: list[str] = Field(default_factory=list)
    answer_options: list[str] = Field(default_factory=list)  # варианты ответа для ускорения
    blocking: bool = False
    context: str = ""                                    # цитата из ТЗ, к которой относится вопрос
    impact: str = ""                                     # что сломается, если ответа не будет
    default_assumption: str = ""                         # что закладываем, если ответа нет до старта спринта
    recommended_option: int | None = None                # индекс рекомендуемого варианта (с 0)
    lint: list[str] = Field(default_factory=list)        # замечания линтера вопросов (пусто — прошёл чеклист)

    @field_validator("recommended_option", mode="before")
    @classmethod
    def _norm_rec(cls, v: Any) -> Any:
        if v is None or isinstance(v, bool):
            return None
        try:
            v = int(str(v).strip())
        except (TypeError, ValueError):
            return None
        return v if v >= 0 else None

    @field_validator("answer_options", "related_findings", mode="before")
    @classmethod
    def _norm_str_list(cls, v: Any) -> Any:
        return to_str_list(v)

    @field_validator("impact", "default_assumption", "why", "context", "addressee", mode="before")
    @classmethod
    def _norm_text(cls, v: Any) -> Any:
        if v is None:
            return ""
        if isinstance(v, list):
            return "; ".join(str(x) for x in v)
        return str(v)

    @field_validator("priority", mode="before")
    @classmethod
    def _norm_prio(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip().upper()
            if v in ("CRITICAL", "HIGH", "BLOCKER"):
                return "P0"
            if v in ("MEDIUM", "MAJOR"):
                return "P1"
            if v in ("LOW", "MINOR"):
                return "P2"
        return v


class Requirement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = ""
    text: str
    kind: str = "functional"   # functional | nonfunctional | security | constraint | business
    line: int | None = None


class DocumentModel(BaseModel):
    """Нормализованное представление ТЗ — результат работы Reader."""
    model_config = ConfigDict(extra="ignore")

    title: str = "Техническое задание"
    summary: str = ""
    product_type: str = ""
    domains: list[str] = Field(default_factory=list)       # transfers, cards, kyc, lending, mobile...
    actors: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    integrations: list[str] = Field(default_factory=list)
    data_categories: list[str] = Field(default_factory=list)  # ПДн, данные карт, биометрия...
    sections: list[str] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)

    @field_validator("domains", "actors", "features", "integrations", "data_categories", "sections", mode="before")
    @classmethod
    def _norm_str_lists(cls, v: Any) -> Any:
        return to_str_list(v)


class ControlCheck(BaseModel):
    """Строка матрицы соответствия финтех/ИБ-контролям."""
    model_config = ConfigDict(extra="ignore")
    control_id: str
    title: str = ""
    area: str = ""
    regulation: str = ""
    status: str = "missing"    # covered | partial | missing | not_applicable
    evidence: list[str] = Field(default_factory=list)
    lines: list[int] = Field(default_factory=list)
    comment: str = ""

    @field_validator("status", mode="before")
    @classmethod
    def _norm_status(cls, v: Any) -> Any:
        return normalize_control_status(v)

    @field_validator("evidence", mode="before")
    @classmethod
    def _norm_evidence(cls, v: Any) -> Any:
        return to_str_list(v)

    @field_validator("lines", mode="before")
    @classmethod
    def _norm_lines(cls, v: Any) -> Any:
        return to_int_list(v)


CONTROL_STATUSES = ("covered", "partial", "missing", "not_applicable")
_STATUS_ALIASES = {
    "covered": "covered", "покрыто": "covered", "покрыт": "covered", "full": "covered", "fully_covered": "covered",
    "fully": "covered", "описано": "covered", "есть": "covered", "ok": "covered", "implemented": "covered",
    "partial": "partial", "partially": "partial", "partially_covered": "partial", "частично": "partial",
    "частичное": "partial", "частичный": "partial", "неполно": "partial", "incomplete": "partial",
    "missing": "missing", "отсутствует": "missing", "absent": "missing", "none": "missing", "нет": "missing",
    "не_описано": "missing", "not_covered": "missing", "uncovered": "missing", "gap": "missing",
    "not_applicable": "not_applicable", "n/a": "not_applicable", "na": "not_applicable", "н/п": "not_applicable",
    "не_применимо": "not_applicable", "неприменимо": "not_applicable", "inapplicable": "not_applicable",
    "irrelevant": "not_applicable",
}


def normalize_control_status(v: Any) -> str:
    """Приводит статус контроля к одному из CONTROL_STATUSES; нераспознанное → 'unknown'
    (агент сам решает, чем его заменить, — обычно статусом пред-проверки)."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return "missing"
    key = re.sub(r"[\s\-]+", "_", str(v).strip().lower()).strip("_.")
    return _STATUS_ALIASES.get(key, "unknown")


class DimensionScore(BaseModel):
    key: str
    name: str
    score: int
    weight: float
    issues: dict[str, int] = Field(default_factory=dict)
    comment: str = ""


class AgentRun(BaseModel):
    agent: str
    title: str
    stage: int
    mode: str = "rules"          # llm | rules | fallback | hybrid
    status: str = "ok"           # ok | error | skipped
    duration_s: float = 0.0
    produced: int = 0
    note: str = ""
    tokens: int = 0


class RejectedRecord(BaseModel):
    """Находка, которую Verifier не пропустил в отчёт, — с человекочитаемой причиной."""
    title: str
    agent: str = ""
    reason: str = ""
    best_line: int | None = None       # ближайшая строка ТЗ, найденная при поиске цитаты
    best_score: float = 0.0            # степень совпадения лучшей цитаты (0..1)
    revision: int = 0                  # 1 — находка уже исправлялась агентом по замечанию Verifier
    outcome: str = ""                  # чем закончилась обратная связь (critic loop)


class MergeRecord(BaseModel):
    kept_id: str = ""                  # ID итоговой находки (F-xx)
    kept_title: str = ""
    merged_title: str = ""
    from_agent: str = ""
    reason: str = ""


class DowngradeRecord(BaseModel):
    id: str = ""
    title: str = ""
    agent: str = ""
    from_severity: str = ""
    to_severity: str = ""
    reason: str = ""
    hit_lines: list[int] = Field(default_factory=list)


class VerifierStats(BaseModel):
    input_findings: int = 0
    merged_duplicates: int = 0
    rejected_hallucinations: int = 0
    downgraded: int = 0
    output_findings: int = 0
    rejected: list[RejectedRecord] = Field(default_factory=list)
    merges: list[MergeRecord] = Field(default_factory=list)
    downgrades: list[DowngradeRecord] = Field(default_factory=list)
    # обратная связь Critic → аналитик (только LLM-режим, не более одного раунда)
    revision_returned: int = 0
    revision_restored: int = 0
    revision_withdrawn: int = 0
    restored: list[RejectedRecord] = Field(default_factory=list)


class AgentMessage(BaseModel):
    """Явное сообщение между агентами при передаче результата (для трассировки и UI)."""
    sender: str
    receiver: str
    kind: str
    count: int = 0
    summary: str = ""
    ts: float = 0.0                    # секунды от начала запуска


class QuestionQuality(BaseModel):
    """Измеримое качество уточняющих вопросов (чеклист линтера + покрытие находок)."""
    total: int = 0
    lint_passed: int = 0
    with_options: float = 0.0          # доля вопросов с ≥2 содержательными вариантами
    closed_p0p1: float = 0.0           # доля закрытых вопросов среди P0/P1
    grounded: float = 0.0              # доля вопросов с цитатой ТЗ или привязкой к находке со строкой
    with_default: float = 0.0          # доля вопросов с допущением по умолчанию
    critical_covered: str = "0/0"
    must_cover_covered: str = "0/0"    # критичные + противоречия + опасные требования + TBD
    findings_covered: str = "0/0"
    overflow: int = 0                  # вопросы сверх лимита встречи («Дополнительные вопросы»)
    repaired: int = 0                  # вопросы, исправленные одним LLM-вызовом по замечаниям линтера
    agenda_minutes: int = 0            # оценка длительности встречи (P0 — 5 мин, P1 — 3, P2 — 1)


class Report(BaseModel):
    document_name: str
    document: DocumentModel
    overall_score: int
    verdict: str
    verdict_level: str            # ready | needs_work | not_ready
    summary: str
    strengths: list[str] = Field(default_factory=list)
    dimensions: list[DimensionScore] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    controls: list[ControlCheck] = Field(default_factory=list)
    trace: list[AgentRun] = Field(default_factory=list)
    verifier: VerifierStats = Field(default_factory=VerifierStats)
    mode: str = "rules"
    model: str = ""
    total_duration_s: float = 0.0
    total_tokens: int = 0
    line_count: int = 0
    template_check: list[dict] = Field(default_factory=list)   # соответствие шаблону ТЗ
    manual_review_hours: float = 0.0                            # оценка времени ручного ревью (см. scorer)
    messages: list[AgentMessage] = Field(default_factory=list)  # сообщения агент → агент
    question_quality: QuestionQuality = Field(default_factory=QuestionQuality)
    extra_questions: list[Question] = Field(default_factory=list)  # вопросы сверх лимита одной встречи

    def severity_counts(self) -> dict[str, int]:
        out = {s.value: 0 for s in Severity}
        for f in self.findings:
            out[f.severity.value] += 1
        return out

    def control_coverage(self) -> tuple[int, int, int]:
        """(покрыто, частично, применимых всего)."""
        applicable = [c for c in self.controls if c.status != "not_applicable"]
        covered = sum(1 for c in applicable if c.status == "covered")
        partial = sum(1 for c in applicable if c.status == "partial")
        return covered, partial, len(applicable)
