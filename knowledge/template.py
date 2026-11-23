"""Проверка ТЗ на соответствие шаблону (например, шаблону организаторов хакатона или
корпоративному шаблону банка).

Из шаблона извлекаются заголовки разделов; раздел считается присутствующим в ТЗ,
если в ТЗ есть похожий заголовок (нечёткое сравнение) или большинство ключевых слов
заголовка встречаются в одном заголовке ТЗ. Дополнительно ищутся незаполненные
поля-заглушки шаблона: <...>, [заполнить], ____.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from models import Category, Finding, Severity
from parsing import Document

_HEADING = re.compile(r"^\s*(#{1,4})\s+(.+?)\s*#*\s*$|^\s*(\d+(?:\.\d+){0,2})\.?\s+([А-ЯA-ZЁ][^.!?]{2,90})$")
_NUM = re.compile(r"^\s*(\d+(\.\d+)*\.?|[IVX]+\.)\s*")
_STOP = {"требования", "требование", "раздел", "системе", "системы", "система", "описание", "общие", "сведения",
         "проекта", "данных", "приложение", "документа", "других", "прочие"}
# <поле>: «<» не после буквы/цифры/оператора, внутри — сначала буква, в конце — не пробел и не «=»;
# так неравенства («p95 < 2 сек при > 100 RPS», «x<y и z>w», «<= 5») не считаются заглушками,
# а ссылки, теги и закрывающие теги (<https://…>, <br>, </p>) исключены отдельно.
PLACEHOLDER = re.compile(
    r"(?<![\w<>=])<(?=[A-Za-zА-Яа-яЁё])(?!https?:|br\b|/)[^<>\n]{1,58}[^\s<>=]>(?!\s*\d)"
    r"|\[\s*(заполн|указ|впис|описат|tbd)[^\]]*\]|_{4,}|\.{6,}|\{\{[^}]+\}\}", re.I)


@dataclass
class TemplateSection:
    id: str
    title: str
    level: int


@dataclass
class TemplateMatch:
    section: TemplateSection
    present: bool
    matched: str = ""
    line: int | None = None
    score: float = 0.0


def _norm(s: str) -> str:
    s = _NUM.sub("", s.lower().replace("ё", "е"))
    s = re.sub(r"[«»\"'()\[\]:;,.!?*_#`]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _stems(s: str) -> set[str]:
    return {w[:6] for w in _norm(s).split() if len(w) >= 4 and w not in _STOP}


def headings(doc: Document, max_level: int = 3) -> list[tuple[int, int, str]]:
    """[(номер строки, уровень, текст)] — заголовки markdown или нумерованные."""
    out = []
    for i, line in enumerate(doc.lines, start=1):
        m = _HEADING.match(line)
        if not m:
            continue
        if m.group(1):
            level, text = len(m.group(1)), m.group(2)
        else:
            level, text = m.group(3).count(".") + 1, m.group(4)
        if level <= max_level and len(text.strip()) >= 3:
            out.append((i, level, text.strip()))
    return out


def sections_from_template(template: Document) -> list[TemplateSection]:
    hs = headings(template)
    # первый заголовок первого уровня — обычно название документа, а не раздел
    if hs and hs[0][1] == 1 and sum(1 for h in hs if h[1] == 1) == 1:
        hs = hs[1:]
    return [TemplateSection(id=f"TPL-{n:02d}", title=_NUM.sub("", t).strip(), level=lvl)
            for n, (_, lvl, t) in enumerate(hs, start=1)]


def match_template(doc: Document, sections: list[TemplateSection]) -> list[TemplateMatch]:
    target = headings(doc, max_level=4)
    out = []
    for s in sections:
        best, best_h = 0.0, None
        sst = _stems(s.title)
        for ln, _, text in target:
            r = SequenceMatcher(None, _norm(s.title), _norm(text)).ratio()
            if sst:
                cov = len(sst & _stems(text)) / len(sst)
                r = max(r, cov * 0.95)
            if r > best:
                best, best_h = r, (ln, text)
        present = best >= 0.6
        out.append(TemplateMatch(s, present, best_h[1] if best_h else "", best_h[0] if best_h else None, round(best, 2)))
    return out


def template_findings(doc: Document, template: Document) -> tuple[list[Finding], list[TemplateMatch]]:
    sections = sections_from_template(template)
    matches = match_template(doc, sections)
    findings = []
    for m in matches:
        if m.present:
            continue
        sev = Severity.major if m.section.level <= 2 else Severity.minor
        findings.append(Finding(
            agent="completeness", category=Category.completeness, severity=sev,
            title=f"Нет раздела из шаблона: {m.section.title}",
            description=f"Шаблон требует раздел «{m.section.title}», но в ТЗ не найден соответствующий заголовок"
                        + (f" (ближайший: «{m.matched}», сходство {m.score:.0%})" if m.matched else "") + ".",
            recommendation=f"Добавить раздел «{m.section.title}» в соответствии с шаблоном.",
            question=f"Раздел «{m.section.title}» обязателен по шаблону — кто и когда его заполнит?",
            control_id=m.section.id, confidence=0.8,
        ))
    for i, line in enumerate(doc.lines, start=1):
        ph = PLACEHOLDER.search(line)
        if ph:
            findings.append(Finding(
                agent="completeness", category=Category.completeness, severity=Severity.major,
                title="Незаполненное поле шаблона",
                description=f"В L{i} осталась заглушка шаблона «{ph.group(0)}» — раздел не заполнен.",
                evidence=[line.strip()[:200]], lines=[i],
                recommendation="Заполнить поле конкретным значением или явно указать «не применимо» с обоснованием.",
                question=f"Чем заполнить поле «{ph.group(0)}» в L{i}?", confidence=0.85,
            ))
    return findings, matches
