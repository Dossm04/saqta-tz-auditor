"""Reader — превращает ТЗ в нормализованную структуру (DocumentModel)."""
from __future__ import annotations

import re

from agents.base import TZ_NOTE, Agent, Context, load_prompt
from knowledge.domains import DOMAIN_LABELS, detect_domains
from models import DocumentModel, Requirement

_REQ = re.compile(r"\b(должен|должна|должно|должны|необходимо|требуется|обязан\w*|обеспечива\w*|shall|must)\b", re.I)
_REQ_ID = re.compile(r"\b((?:FR|NFR|ФТ|НФТ|ТР|REQ|BR|SR|US)[-.\s]?\d+(?:\.\d+)*)", re.I)
_HEADING = re.compile(r"^\s*(#{1,6})\s+(.+)$|^\s*(\d+(?:\.\d+)*)\.?\s+([А-ЯA-ZЁ][^.]{2,80})$")

_INTEGRATIONS = {
    r"\bабс\b|core ?banking": "АБС (core banking)", r"процессинг": "Процессинг карт", r"\bгбд\b|гбд фл": "ГБД ФЛ",
    r"sms|смс": "SMS-шлюз", r"push": "Push-уведомления", r"\baml\b|финмониторинг|под/фт": "AML-система",
    r"антифрод|anti-?fraud": "Антифрод", r"кредитн\w* бюро|\bпкб\b|\bгкб\b": "Кредитное бюро",
    r"эгов|egov": "eGov", r"visa|mastercard": "Международные платёжные системы", r"\bмсб\b|межбанк": "Межбанковская система переводов",
    r"биометр": "Сервис биометрии", r"1с\b|1c\b": "1С",
}
_DATA = {
    r"\bиин\b": "ИИН", r"персональн\w* данн|фио|адрес|телефон": "Персональные данные",
    r"номер\w* карт|\bpan\b|cvv|cvc": "Данные платёжных карт", r"биометр|селфи|лиц[оа]": "Биометрия",
    r"геолокац|местоположен": "Геолокация", r"паспорт|удостоверени": "Документы, удостоверяющие личность",
    r"перевод|платеж|транзакц|баланс": "Финансовые операции",
}
_ACTORS = {
    r"\bклиент": "Клиент", r"администратор": "Администратор", r"операционист|оператор": "Операционист",
    r"комплаенс": "Комплаенс-офицер", r"колл-?центр|поддержк": "Сотрудник поддержки", r"мерчант|продав": "Мерчант",
    r"получател": "Получатель перевода", r"менеджер": "Менеджер",
}


def clip_words(s: str, n: int) -> str:
    """Обрезает строку до n символов по границе слова, чтобы не разрезать число (ИИН, PAN, телефон):
    обрезок номера уже не узнаётся детекторами ПДн режима конфиденциальности."""
    if len(s) <= n:
        return s
    cut = s[:n]
    if not s[n].isspace():
        ws = max(cut.rfind(" "), cut.rfind("\t"))
        if ws > 0:
            cut = cut[:ws]
        else:                              # одно длинное «слово» — хотя бы не оставляем обрезок числа
            cut = re.sub(r"[\d \t\-]+$", "", cut) or cut
    return cut.rstrip()


def valid_requirements(reqs: list[Requirement], doc_lines: list[str]) -> list[Requirement]:
    """Оставляет требования, чья строка существует и непуста (line=None — без привязки, допустимо)."""
    n = len(doc_lines)
    return [r for r in reqs if r.line is None or (1 <= r.line <= n and doc_lines[r.line - 1].strip())]


def merge_models(rules_dm: DocumentModel, llm_dm: DocumentModel,
                 doc_lines: list[str] | None = None) -> DocumentModel:
    """Объединяет модель правил и LLM: домены — объединение (безопаснее проверить лишний контроль,
    чем пропустить), списки — объединение без дублей, пустые поля LLM берутся из правил.

    doc_lines — строки документа: требования LLM со ссылкой на несуществующую или пустую строку
    отбрасываются; если валидных не осталось — берутся требования правил."""
    dm = llm_dm.model_copy(deep=True)
    dm.domains = [d for d in DOMAIN_LABELS if d in set(llm_dm.domains) | set(rules_dm.domains)]
    for attr in ("actors", "integrations", "data_categories", "features"):
        merged = list(dict.fromkeys([*getattr(llm_dm, attr), *getattr(rules_dm, attr)]))
        setattr(dm, attr, merged[:40])
    if not dm.sections:
        dm.sections = rules_dm.sections
    if doc_lines is not None:
        dm.requirements = valid_requirements(dm.requirements, doc_lines)
    if not dm.requirements:
        dm.requirements = rules_dm.requirements
    dm.requirements = dm.requirements[:80]
    if not dm.title.strip() or dm.title == "Техническое задание":
        dm.title = rules_dm.title
    if not dm.summary.strip():
        dm.summary = rules_dm.summary
    if not dm.product_type.strip():
        dm.product_type = rules_dm.product_type
    return dm


def _collect(text: str, mapping: dict[str, str]) -> list[str]:
    t = text.lower().replace("ё", "е")
    return list(dict.fromkeys(v for p, v in mapping.items() if re.search(p, t)))


class ReaderAgent(Agent):
    name = "reader"
    title = "Reader"
    role = "Парсит ТЗ в структуру: цель, домены, роли, требования, интеграции, данные"
    stage = 1

    async def run_llm(self, ctx: Context) -> tuple[int, str]:
        """LLM-обогащение модели документа. В конвейере идёт параллельно со стадией 2:
        мгновенная модель правил уже лежит в ctx.doc_model, LLM дополняет её для вопросов, оценки и отчёта."""
        rules_dm = ctx.doc_model if ctx.doc_model.sections or ctx.doc_model.requirements else self._rules_model(ctx)
        user = f"{TZ_NOTE}\n{ctx.tz_block}"
        if len(ctx.doc.numbered()) > ctx.cfg.max_doc_chars:   # длинное ТЗ: оглавление целиком, текст выше обрезан
            user += f"\n\n## Оглавление всего ТЗ (текст выше обрезан)\n{ctx.doc.outline()}"
        dm = await self.ask(ctx, load_prompt("reader"), user, DocumentModel)
        dm = merge_models(rules_dm, dm, ctx.doc.lines)
        added = [d for d in dm.domains if d not in rules_dm.domains]
        ctx.doc_model, ctx.domains = dm, dm.domains
        return len(dm.requirements), (f"LLM-обогащение: домены {', '.join(dm.domains) or '—'}"
                                      + (f" (+{', '.join(added)})" if added else ""))

    def run_rules(self, ctx: Context) -> tuple[int, str]:
        dm = self._rules_model(ctx)
        ctx.doc_model, ctx.domains = dm, dm.domains
        return len(dm.requirements), f"домены: {', '.join(dm.domains) or '—'}"

    def _rules_model(self, ctx: Context) -> DocumentModel:
        doc = ctx.doc
        sections, title = [], ""
        for line in doc.lines:
            m = _HEADING.match(line)
            if m:
                h = (m.group(2) or m.group(4) or "").strip()
                if h:
                    sections.append(h)
                    if not title and (m.group(1) or "") == "#":
                        title = h
        title = title or (sections[0] if sections else doc.name)
        title = re.sub(r"^(техническое задание|тз)\s*(на\s+)?[:—-]?\s*", "", title, flags=re.I).strip() or title
        reqs: list[Requirement] = []
        for i, line in enumerate(doc.lines, start=1):
            s = line.strip()
            mid = _REQ_ID.match(s.lstrip("-*• "))
            if len(s) < 15 or not (_REQ.search(s) or mid):
                continue
            low = s.lower()
            kind = ("security" if re.search(r"безопас|шифр|аутент|парол|доступ", low) else
                    "nonfunctional" if re.search(r"доступност|время|производ|нагрузк|sla|%", low) else "functional")
            reqs.append(Requirement(id=mid.group(1) if mid else f"R-{len(reqs) + 1}", text=clip_words(s.lstrip("-*• "), 220),
                                    kind=kind, line=i))
        domains = detect_domains(doc.text)
        return DocumentModel(
            title=title.strip("# "), product_type=", ".join(DOMAIN_LABELS[d] for d in domains[:3]),
            summary=f"ТЗ «{title.strip('# ')}»: {doc.non_empty_count} непустых строк, {len(reqs)} требований-кандидатов. "
                    f"Затронутые области: {', '.join(DOMAIN_LABELS[d] for d in domains) or 'не определены'}.",
            domains=domains, actors=_collect(doc.text, _ACTORS), integrations=_collect(doc.text, _INTEGRATIONS),
            data_categories=_collect(doc.text, _DATA), sections=sections[:60], requirements=reqs[:120],
            features=[clip_words(r.text, 90) for r in reqs if r.kind == "functional"][:20],
        )
