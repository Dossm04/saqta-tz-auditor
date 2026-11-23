"""Reader — превращает ТЗ в нормализованную структуру (DocumentModel)."""
from __future__ import annotations

import re

from agents.base import Agent, Context, load_prompt
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


def _collect(text: str, mapping: dict[str, str]) -> list[str]:
    t = text.lower().replace("ё", "е")
    return list(dict.fromkeys(v for p, v in mapping.items() if re.search(p, t)))


class ReaderAgent(Agent):
    name = "reader"
    title = "Reader"
    role = "Парсит ТЗ в структуру: цель, домены, роли, требования, интеграции, данные"
    stage = 1

    async def run_llm(self, ctx: Context) -> tuple[int, str]:
        dm = await self.ask(ctx, load_prompt("reader"), ctx.numbered_doc, DocumentModel)
        rules_dm = self._rules_model(ctx)
        # объединяем домены LLM и правил: безопаснее проверить лишний контроль, чем пропустить
        dm.domains = [d for d in DOMAIN_LABELS if d in set(dm.domains) | set(rules_dm.domains)]
        if not dm.sections:
            dm.sections = rules_dm.sections
        dm.requirements = dm.requirements[:80]
        ctx.doc_model, ctx.domains = dm, dm.domains
        return len(dm.requirements), f"домены: {', '.join(dm.domains) or '—'}"

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
            reqs.append(Requirement(id=mid.group(1) if mid else f"R-{len(reqs) + 1}", text=s.lstrip("-*• ")[:220],
                                    kind=kind, line=i))
        domains = detect_domains(doc.text)
        return DocumentModel(
            title=title.strip("# "), product_type=", ".join(DOMAIN_LABELS[d] for d in domains[:3]),
            summary=f"ТЗ «{title.strip('# ')}»: {doc.non_empty_count} непустых строк, {len(reqs)} требований-кандидатов. "
                    f"Затронутые области: {', '.join(DOMAIN_LABELS[d] for d in domains) or 'не определены'}.",
            domains=domains, actors=_collect(doc.text, _ACTORS), integrations=_collect(doc.text, _INTEGRATIONS),
            data_categories=_collect(doc.text, _DATA), sections=sections[:60], requirements=reqs[:120],
            features=[r.text[:90] for r in reqs if r.kind == "functional"][:20],
        )
