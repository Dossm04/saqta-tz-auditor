"""Структурное чтение PDF: восстанавливает разметку, на которую опираются нарезка требований и проверки.

Простое `extract_text()` теряет заголовки, рвёт фразы на строки и перемешивает ячейки таблиц.
Здесь:
  * заголовки определяются по размеру и жирности шрифта → строки `#`, `##`, …;
  * таблицы извлекаются `find_tables()` → строки `| a | b |` с разделителем после шапки;
  * перенесённые строки абзаца склеиваются, пункты списков и нумерация начинают новый абзац;
  * моноширинный текст (код, JSON, curl) оборачивается в ```;
  * для каждой строки результата запоминается номер страницы.
"""
from __future__ import annotations

import io
import re
from collections import Counter
from statistics import median

_ITEM_START = re.compile(r"^(?:[•▪●◦\-–*]\s+|\d+(?:\.\d+)*[.)]\s+|[A-ZА-ЯЁ]{1,5}[-‑–]\d+(?:\.\d+)*[.)]?\s)")
_MONO = re.compile(r"courier|mono|consolas|menlo|code", re.I)
_BOLD = re.compile(r"bold|black|heavy|semibold|demi", re.I)


def _cell(v) -> str:
    return re.sub(r"\s+", " ", (v or "").replace("\n", " ")).strip()


def _in_bbox(obj, boxes) -> bool:
    x = (obj["x0"] + obj["x1"]) / 2
    y = (obj["top"] + obj["bottom"]) / 2
    return any(b[0] - 1 <= x <= b[2] + 1 and b[1] - 1 <= y <= b[3] + 1 for b in boxes)


def _lines_from_words(words: list[dict], tol: float = 2.5) -> list[dict]:
    """Группировка слов в строки по вертикали."""
    lines: list[dict] = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if lines and abs(w["top"] - lines[-1]["top"]) <= tol:
            lines[-1]["words"].append(w)
        else:
            lines.append({"top": w["top"], "words": [w]})
    out = []
    for ln in lines:
        ws = sorted(ln["words"], key=lambda w: w["x0"])
        sizes = [w.get("size", 0) for w in ws]
        fonts = [w.get("fontname", "") for w in ws]
        out.append({
            "top": ln["top"], "bottom": max(w["bottom"] for w in ws), "x0": ws[0]["x0"],
            "text": " ".join(w["text"] for w in ws),
            "size": max(sizes) if sizes else 0,
            "bold": all(_BOLD.search(f or "") for f in fonts),
            "mono": all(_MONO.search(f or "") for f in fonts),
        })
    return out


def _is_heading_like(it: dict, level_of: dict) -> bool:
    text = it["text"].strip()
    return (round(it["size"], 1) in level_of and len(text) < 160) or (
        it["bold"] and len(text) < 120 and not text.endswith((".", ":", ";")))


def read_pdf_structured(data: bytes) -> tuple[str, list[int]]:
    import pdfplumber

    items: list[dict] = []          # строки текста и таблицы в порядке чтения
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.find_tables()
            except Exception:
                tables = []
            boxes = [t.bbox for t in tables]
            words = page.filter(lambda o: o.get("object_type") != "char" or not _in_bbox(o, boxes)).extract_words(
                extra_attrs=["size", "fontname"], keep_blank_chars=False, use_text_flow=False)
            for ln in _lines_from_words(words):
                ln["page"] = pno
                ln["kind"] = "line"
                items.append(ln)
            for t in tables:
                try:
                    rows = [[_cell(c) for c in r] for r in t.extract() if any(_cell(c) for c in r)]
                except Exception:
                    continue
                if rows:
                    items.append({"kind": "table", "page": pno, "top": t.bbox[1], "rows": rows})
            # порядок внутри страницы — сверху вниз
            page_items = [i for i in items if i["page"] == pno]
            page_items.sort(key=lambda i: i["top"])
            items = [i for i in items if i["page"] != pno] + page_items

    text_lines = [i for i in items if i["kind"] == "line"]
    if not text_lines and not any(i["kind"] == "table" for i in items):
        return "", []
    sizes = Counter(round(i["size"], 1) for i in text_lines for _ in i["text"].split())
    body = sizes.most_common(1)[0][0] if sizes else 10.0
    head_sizes = sorted({round(i["size"], 1) for i in text_lines if i["size"] >= body * 1.12}, reverse=True)
    level_of = {s: min(k + 1, 4) for k, s in enumerate(head_sizes)}
    bold_level = min(len(head_sizes) + 1, 4)

    out: list[str] = []
    pages: list[int] = []

    def emit(s: str, page: int) -> None:
        out.append(s)
        pages.append(page)

    para: list[str] = []
    para_page = 1
    prev: dict | None = None
    in_code = False
    gaps = [b["top"] - a["bottom"] for a, b in zip(text_lines, text_lines[1:]) if a["page"] == b["page"] and b["top"] > a["bottom"]]
    line_gap = median(gaps) if gaps else 3.0

    def flush() -> None:
        nonlocal para
        if para:
            joined = ""
            for part in para:
                if joined.endswith("-") and not joined.endswith(" -"):
                    joined = joined[:-1] + part           # перенос со знаком дефиса
                else:
                    joined = (joined + " " + part).strip()
            emit(joined, para_page)
            para = []

    last_table: dict | None = None      # {"header": [...], "rows_idx": [индексы строк в out], "orphans": [...]}
    for it in items:
        if it["kind"] == "table":
            flush()
            if in_code:
                emit("```", it["page"])
                in_code = False
            rows = it["rows"]
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
            # продолжение таблицы на новой странице: повторённая шапка и разорванная строка
            if last_table and rows[0] == last_table["header"] and it["page"] > last_table["page"]:
                rows = rows[1:]
                # строки текста между частями таблицы — обрывки разорванной строки; удаляем их
                orphans: list[str] = []
                while len(out) > last_table["end"] and not out[-1].startswith(("|", "#")):
                    if out[-1].strip():
                        orphans.insert(0, out[-1].strip())
                    out.pop()
                    pages.pop()
                while out and out[-1] == "":
                    out.pop()
                    pages.pop()
                if rows and rows[0][0] and not any(rows[0][1:]):
                    tail = rows[0][0]
                    rows = rows[1:]
                    if orphans:
                        # начало разорванной строки попало в текст вне таблицы: значения других ячеек
                        # («Да / Нет») вырезаем, первую ячейку восстанавливаем целиком
                        head = re.sub(r"\s*Да\s*/\s*Нет(?:\s*/\s*Не требуется)?\s*", " ", " ".join(orphans)).strip()
                        rows.insert(0, [f"{head} {tail}".strip()] + [""] * (width - 1))
                    elif last_table["rows_idx"]:
                        idx = last_table["rows_idx"][-1]
                        cells = [c.strip() for c in out[idx].strip("|").split("|")]
                        cells[0] = (cells[0] + " " + tail).strip()
                        out[idx] = "| " + " | ".join(cells) + " |"
                for r in rows:
                    emit("| " + " | ".join(r) + " |", it["page"])
                    last_table["rows_idx"].append(len(out) - 1)
                last_table["page"] = it["page"]
                last_table["end"] = len(out)
                emit("", it["page"])
                prev = None
                continue
            emit("", it["page"])
            rows_idx = []
            for k, r in enumerate(rows):
                emit("| " + " | ".join(r) + " |", it["page"])
                if k == 0:
                    emit("|" + "---|" * width, it["page"])
                else:
                    rows_idx.append(len(out) - 1)
            last_table = {"header": rows[0], "rows_idx": rows_idx, "page": it["page"], "end": len(out)}
            emit("", it["page"])
            prev = None
            continue
        if it["kind"] == "line" and _is_heading_like(it, level_of):
            last_table = None

        text = it["text"].strip()
        if not text:
            continue
        size = round(it["size"], 1)
        is_heading = (size in level_of and len(text) < 160) or (it["bold"] and len(text) < 120 and not text.endswith((".", ":", ";")))
        new_block = (prev is None or it["page"] != prev["page"] or it["top"] - prev["bottom"] > max(line_gap * 1.8, it["size"] * 0.6))

        if it["mono"] and not is_heading:
            flush()
            if not in_code:
                emit("```", it["page"])
                in_code = True
            emit(text, it["page"])
            prev = it
            continue
        if in_code:
            emit("```", it["page"])
            in_code = False

        if is_heading:
            flush()
            lvl = level_of.get(size, bold_level)
            emit("", it["page"])
            emit("#" * lvl + " " + text, it["page"])
            prev = it
            continue
        if new_block or _ITEM_START.match(text) or (prev is not None and prev.get("bold") and not para):
            flush()
            if new_block and out and out[-1] != "":
                emit("", it["page"])
            para_page = it["page"]
        if not para:
            para_page = it["page"]
        para.append(text)
        prev = it
    flush()
    if in_code:
        emit("```", pages[-1] if pages else 1)
    # убрать подряд идущие пустые строки
    cleaned, cpages = [], []
    for s, p in zip(out, pages):
        if s == "" and cleaned and cleaned[-1] == "":
            continue
        cleaned.append(s)
        cpages.append(p)
    return "\n".join(cleaned), cpages
