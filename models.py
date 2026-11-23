"""Pydantic-схемы: находки, вопросы, модель документа, итоговый отчёт.

Все агенты обмениваются строго типизированными объектами — это делает
отчёт детерминированным и позволяет проверять вывод LLM.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
        out = []
        for x in v or []:
            try:
                out.append(int(str(x).lstrip("Ll")))
            except ValueError:
                continue
        return out

    @field_validator("evidence", mode="before")
    @classmethod
    def _norm_evidence(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [v] if v.strip() else []
        return [str(x) for x in (v or []) if str(x).strip()]


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
        v = str(v or "missing").strip().lower()
        return {"покрыто": "covered", "частично": "partial", "отсутствует": "missing",
                "n/a": "not_applicable", "na": "not_applicable"}.get(v, v)


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


class VerifierStats(BaseModel):
    input_findings: int = 0
    merged_duplicates: int = 0
    rejected_hallucinations: int = 0
    downgraded: int = 0
    output_findings: int = 0


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
