"""Чтение ТЗ из .md / .txt / .docx / .pdf в нумерованные строки.

Нумерация строк (L1, L2, ...) — основа «заземления»: каждый агент обязан
ссылаться на строки и дословные цитаты, а Verifier проверяет их наличие.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path


def _norm(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[«»\"'`“”„]", "", s)
    s = re.sub(r"[*_#>|]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Document:
    name: str
    text: str
    lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.lines:
            self.lines = self.text.splitlines()
        self._norm_lines = [_norm(l) for l in self.lines]
        self._norm_full = _norm(" ".join(self.lines))

    # --- представление для LLM -------------------------------------------------
    def numbered(self, max_chars: int | None = None) -> str:
        out, total = [], 0
        for i, line in enumerate(self.lines, start=1):
            if not line.strip():
                continue
            row = f"L{i}: {line.rstrip()}"
            total += len(row) + 1
            if max_chars and total > max_chars:
                out.append(f"... [документ обрезан: показано до L{i - 1} из {len(self.lines)}]")
                break
            out.append(row)
        return "\n".join(out)

    def line(self, n: int) -> str:
        return self.lines[n - 1] if 1 <= n <= len(self.lines) else ""

    @property
    def non_empty_count(self) -> int:
        return sum(1 for l in self.lines if l.strip())

    # --- проверка цитат ------------------------------------------------------
    def locate_quote(self, quote: str) -> tuple[int | None, float]:
        """Ищет цитату в документе. Возвращает (номер строки, степень совпадения 0..1)."""
        q = _norm(quote).strip(" .…")
        if len(q) < 3:
            return None, 0.0
        # 1) точное вхождение в строку
        for i, nl in enumerate(self._norm_lines, start=1):
            if q in nl:
                return i, 1.0
        # 2) цитата через границу строк
        if q in self._norm_full:
            pos = self._norm_full.find(q)
            acc = 0
            for i, nl in enumerate(self._norm_lines, start=1):
                acc += len(nl) + 1
                if acc > pos:
                    return i, 0.98
        # 3) нечёткое сравнение (LLM слегка перефразировал)
        best_i, best = None, 0.0
        for i, nl in enumerate(self._norm_lines, start=1):
            if not nl:
                continue
            if len(nl) >= len(q):
                # скользящее окно по длине цитаты
                step = max(1, len(q) // 4)
                for start in range(0, len(nl) - len(q) + 1, step):
                    r = SequenceMatcher(None, q, nl[start:start + len(q)]).ratio()
                    if r > best:
                        best, best_i = r, i
            else:
                r = SequenceMatcher(None, q, nl).ratio() * (len(nl) / len(q)) ** 0.5
                if r > best:
                    best, best_i = r, i
        return best_i, round(best, 3)

    def search(self, pattern: str | re.Pattern) -> list[int]:
        rx = re.compile(pattern, re.IGNORECASE) if isinstance(pattern, str) else pattern
        return [i for i, l in enumerate(self.lines, start=1) if rx.search(l.replace("ё", "е").replace("Ё", "Е"))]


# --- загрузчики ---------------------------------------------------------------

def _read_docx(data: bytes) -> str:
    import docx  # python-docx

    d = docx.Document(io.BytesIO(data))
    out: list[str] = []
    body = d.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = docx.text.paragraph.Paragraph(child, d)
            text = p.text.strip()
            if not text:
                out.append("")
                continue
            style = (p.style.name or "").lower() if p.style is not None else ""
            m = re.search(r"(heading|заголовок)\s*(\d)", style)
            if m:
                out.append("#" * int(m.group(2)) + " " + text)
            elif "list" in style or "спис" in style:
                out.append("- " + text)
            else:
                out.append(text)
        elif tag == "tbl":
            t = docx.table.Table(child, d)
            for row in t.rows:
                cells = []
                for c in row.cells:
                    ct = c.text.strip().replace("\n", " ")
                    if not cells or cells[-1] != ct:  # объединённые ячейки
                        cells.append(ct)
                out.append("| " + " | ".join(cells) + " |")
            out.append("")
    return "\n".join(out)


def _read_pdf(data: bytes) -> str:
    import pdfplumber

    pages = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1251", "utf-16"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_bytes(data: bytes, filename: str) -> Document:
    ext = Path(filename).suffix.lower()
    if ext == ".docx":
        text = _read_docx(data)
    elif ext == ".pdf":
        text = _read_pdf(data)
    else:
        text = _decode(data)
    text = text.replace("\r\n", "\n").replace("\t", "    ")
    return Document(name=Path(filename).name, text=text)


def load_path(path: str | Path) -> Document:
    p = Path(path)
    return load_bytes(p.read_bytes(), p.name)


def load_text(text: str, name: str = "pasted.md") -> Document:
    return Document(name=name, text=text.replace("\r\n", "\n"))
