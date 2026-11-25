"""Нарезка документа на отдельные требования (детерминированно и воспроизводимо).

Единица проверки:
  * пункт списка («Ф-1. …», «1. …», «- …») — с идентификатором из документа, если он есть;
  * абзац раздела с требованиями (НФТ, данные, интеграция, UI, цели…);
  * строка таблицы с требованиями (критерии приёмки, параметры API, общие параметры метода);
  * строка статуса API («200. …»); блоки кода (JSON, curl) прикрепляются к предыдущей единице.

Не являются требованиями: титул и преамбула документа, «Общие сведения», служебные строки
паспорта метода (автор, разработчик, проект), чек-лист ревью и блок согласующих — чек-лист
сохраняется отдельно и заполняется агентом при проверке набора.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from parsing import Document

_HEADING = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*#*\s*$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_BULLET = re.compile(r"^\s*(?:[-*•▪]\s+|\d+[.)]\s+)")
_NUM_ITEM = re.compile(r"^\s*(\d+)[.)]\s+")
_EXPLICIT_ID = re.compile(r"^(?:\*\*)?\s*([A-ZА-ЯЁ]{1,6}(?:-[A-ZА-ЯЁ]{1,6})?[-‑–]\d+(?:\.\d+)*)\.?\s*(?:\*\*)?\s*")
_STATUS_LINE = re.compile(r"^\s*(?:\*\*)?\s*([1-5]\d\d)\.?\s*(?:\*\*)?\s+\S")
_BOLD_LABEL = re.compile(r"^\s*\*\*([^*]{1,40})\*\*\s*$")      # «**404**» — подзаголовок, оформленный жирным
_CONNECTOR = {"или", "либо", "or", "и"}
_MD = re.compile(r"\*\*|__|`")

SKIP_SECTIONS = ("общие сведения", "ревью требований", "блок согласующих", "лист согласования", "история изменений",
                 "содержание", "оглавление")
CHECKLIST_SECTIONS = ("ревью требований", "чек-лист", "чеклист")
META_ROW_KEYS = {"gitlab project", "framework", "author", "developer", "автор", "разработчик", "url type",
                 "наименование задачи", "продукт", "задача в jira"}

KINDS = [  # (подстрока заголовка, вид требования)
    ("нефункциональн", "nonfunctional"), ("функциональн", "functional"), ("критери", "acceptance"),
    ("тестирован", "testing"), ("данн", "data"), ("интеграц", "integration"), ("дизайн", "ui"), ("ui", "ui"),
    ("bpmn", "process"), ("процесс", "process"), ("sequence", "process"), ("архитектур", "architecture"),
    ("схема", "architecture"), ("цел", "business"), ("логика", "api_logic"), ("особые правила", "business_rule"),
    ("правил", "business_rule"), ("парамет", "interface"), ("request", "interface"), ("ответ", "interface"),
    ("статус", "interface"), ("общие параметры", "interface"), ("response", "interface"),
]
KIND_LABEL = {
    "functional": "функциональное", "nonfunctional": "нефункциональное", "acceptance": "критерий приёмки",
    "testing": "тестирование", "data": "данные", "integration": "интеграция", "ui": "интерфейс (UI)",
    "process": "процесс", "architecture": "архитектура", "business": "бизнес-цель", "api_logic": "логика API",
    "business_rule": "бизнес-правило", "interface": "интерфейс API", "other": "прочее",
}


@dataclass
class ReqUnit:
    rid: str
    doc_ref: str                 # идентификатор в документе: «Ф-1» или производный («Логика работы метода, п. 2»)
    section: str                 # путь разделов
    text: str                    # формулировка без разметки (для проверки)
    quote: str                   # исходная строка(и) документа
    lines: list[int]
    kind: str = "other"
    explicit_id: bool = False
    extra: list[str] = field(default_factory=list)    # прикреплённые блоки кода / примеры

    @property
    def short(self) -> str:
        t = self.text
        return t if len(t) <= 120 else t[:117].rsplit(" ", 1)[0] + "…"

    @property
    def kind_label(self) -> str:
        return KIND_LABEL.get(self.kind, self.kind)


@dataclass
class Segmentation:
    units: list[ReqUnit]
    checklist: list[dict]        # строки чек-листа ревью из документа: {"item", "line", "values"}
    title: str
    sections: list[str]

    def by_rid(self) -> dict[str, ReqUnit]:
        return {u.rid: u for u in self.units}

    def unit_for_line(self, line: int) -> ReqUnit | None:
        for u in self.units:
            if line in u.lines:
                return u
        return None


def _clean(s: str) -> str:
    s = _MD.sub("", s)
    s = re.sub(r"^\s*(?:[-*•▪]\s+)", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _kind(path: list[str]) -> str:
    for h in reversed(path):
        hl = h.lower()
        for key, kind in KINDS:
            if key in hl:
                return kind
    return "other"


def segment(doc: Document) -> Segmentation:
    units: list[ReqUnit] = []
    checklist: list[dict] = []
    path: list[tuple[int, str]] = []   # (уровень, заголовок)
    title = ""
    para: list[tuple[int, str]] = []
    table_header: list[str] | None = None
    table_rows_seen = 0
    in_code = False
    code_buf: list[tuple[int, str]] = []
    counters: dict[str, int] = {}
    sections: list[str] = []

    def section_path() -> list[str]:
        return [h for lvl, h in path if lvl > 1 or len(path) == 1 and False]

    def path_str() -> str:
        names = [h for lvl, h in path if lvl >= 2]
        return " › ".join(names)

    def skip_here() -> bool:
        p = path_str().lower()
        return not p or any(s in p for s in SKIP_SECTIONS)

    def is_checklist() -> bool:
        p = path_str().lower()
        return any(s in p for s in CHECKLIST_SECTIONS)

    def last_section() -> str:
        names = [h for lvl, h in path if lvl >= 2]
        return names[-1] if names else ""

    def status_section() -> str:
        names = [h for lvl, h in path if lvl >= 2]
        return next((n for n in names if "статус" in n.lower() or "status" in n.lower()), last_section())

    def add(text: str, quote: str, lines: list[int], ref_suffix: str | None = None) -> None:
        clean = _clean(text)
        if not clean or len(clean) < 3 or clean.lower() in _CONNECTOR:
            return
        m = _EXPLICIT_ID.match(text.strip().lstrip("-*• ").strip())
        explicit = bool(m)
        if m:
            doc_ref = m.group(1).replace("‑", "-").replace("–", "-")
            clean = _clean(text.strip().lstrip("-*• ")[m.end():]) or clean
        elif ref_suffix and ref_suffix.startswith("код "):
            doc_ref = f"{status_section()}, {ref_suffix}"
        else:
            sec = last_section()
            key = sec.lower()
            counters[key] = counters.get(key, 0) + 1
            doc_ref = f"{sec}, {ref_suffix}" if ref_suffix else f"{sec}, абз. {counters[key]}"
        units.append(ReqUnit(rid="", doc_ref=doc_ref, section=path_str(), text=clean, quote=quote.strip(),
                             lines=lines, kind=_kind([h for _, h in path]), explicit_id=explicit))

    def flush_para() -> None:
        nonlocal para
        # вводный абзац части документа («Часть B. REST API: дочерний документ части A…») — контекст, не требование
        if para and re.match(r"^(часть|part|раздел)\s+[\wА-Яа-я]{1,3}\b", last_section().lower()):
            para = []
            return
        if para and not skip_here() and not in_code:
            lines = [n for n, _ in para]
            text = " ".join(t.strip() for _, t in para)
            st = _STATUS_LINE.match(text)
            suffix = f"код {st.group(1)}" if st else None
            add(text, "\n".join(t for _, t in para), lines, suffix)
        para = []

    def attach_code() -> None:
        nonlocal code_buf
        if code_buf and not skip_here():
            code = "\n".join(t for _, t in code_buf)
            if len(units) == units_at_heading[0] and heading_line[0]:
                # пример сразу под заголовком («#### 200. Successful», «**404**») — отдельная единица
                hl, ht = heading_line[0]
                st = re.match(r"^\s*([1-5]\d\d)\b", ht)
                add(f"{ht}: пример ответа {code}" if st else f"{ht}: {code}", ht, [hl] + [n for n, _ in code_buf],
                    f"код {st.group(1)}, пример" if st else "блок 1")
                if units:
                    units[-1].extra.append(code)
            elif units:
                u = units[-1]
                u.extra.append(code)
                u.lines.extend(n for n, _ in code_buf)
        code_buf = []

    units_at_heading = [0]
    heading_line: list[tuple[int, str] | None] = [None]

    for i, raw in enumerate(doc.lines, start=1):
        line = raw.rstrip()
        if line.strip().startswith("```"):
            if in_code:
                in_code = False
                attach_code()
            else:
                flush_para()
                in_code = True
            continue
        if in_code:
            code_buf.append((i, line))
            continue
        h = _HEADING.match(line)
        bl = _BOLD_LABEL.match(line)
        if h or bl:
            flush_para()
            if h:
                lvl, text = len(h.group(1)), _clean(h.group(2))
            else:   # жирная подпись-заголовок на отдельной строке — уровень ниже текущего раздела
                lvl, text = (path[-1][0] + 1 if path else 2), _clean(bl.group(1))
            while path and path[-1][0] >= lvl:
                path.pop()
            path.append((lvl, text))
            table_header = None
            units_at_heading[0] = len(units)
            heading_line[0] = (i, text)
            if lvl == 1 and not title:
                title = text
            elif lvl >= 2:
                sections.append(path_str())
            continue
        if not line.strip():
            flush_para()
            table_header = None
            continue
        if _TABLE_ROW.match(line):
            flush_para()
            if _TABLE_SEP.match(line.replace("|", "|")) or set(line.replace("|", "").strip()) <= set("-: "):
                continue
            cells = _cells(line)
            if table_header is None:
                table_header = cells
                table_rows_seen = 0
                continue
            table_rows_seen += 1
            if is_checklist():
                if cells and cells[0]:
                    checklist.append({"item": _clean(cells[0]), "line": i, "values": cells[1:]})
                continue
            if skip_here():
                continue
            key = _clean(cells[0]) if cells else ""
            if key.lower() in META_ROW_KEYS or not any(c.strip() for c in cells[1:]) and len(cells) > 1 and not key:
                continue
            parts = [f"{_clean(hd)}: {_clean(c)}" for hd, c in zip(table_header, cells) if _clean(c)]
            text = "; ".join(parts)
            kind_path = [h for _, h in path]
            is_param = any(k in " ".join(kind_path).lower() for k in ("парамет", "ответ", "общие параметры", "response"))
            suffix = f"параметр «{key}»" if is_param and key else f"строка {table_rows_seen}"
            add(text, line, [i], suffix)
            continue
        table_header = None
        if _STATUS_LINE.match(line) and any("статус" in h.lower() or "ответ" in h.lower() or "status" in h.lower()
                                            or re.match(r"^[1-5]\d\d\b", h) for _, h in path):
            # «200. Ответ в случае…» — описание кода ответа (а не нумерованный пункт «п. 200»)
            flush_para()
            para = [(i, line)]
            flush_para()
            continue
        if _BULLET.match(line) or _EXPLICIT_ID.match(line.strip()):
            flush_para()
            num = _NUM_ITEM.match(line)
            para = [(i, line)]
            if num and not _EXPLICIT_ID.match(re.sub(r"^\s*\d+[.)]\s+", "", line)):
                # нумерованный пункт без собственного ID → «<раздел>, п. N»
                if not skip_here():
                    add(re.sub(r"^\s*\d+[.)]\s+", "", line), line, [i], f"п. {num.group(1)}")
                para = []
            continue
        para.append((i, line))
    flush_para()
    if in_code:
        attach_code()

    for k, u in enumerate(units, start=1):
        u.rid = f"R-{k:02d}"
    return Segmentation(units=units, checklist=checklist, title=title or doc.name, sections=sections)
