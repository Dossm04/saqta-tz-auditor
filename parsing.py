"""Чтение ТЗ из .md / .txt / .docx / .pdf в нумерованные строки.

Нумерация строк (L1, L2, ...) — основа «заземления»: каждый агент обязан
ссылаться на строки и дословные цитаты, а Verifier проверяет их наличие.
"""
from __future__ import annotations

import bisect
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

_HEADING = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*#*\s*$|^\s*(\d+(?:\.\d+){0,3})\.?\s+([А-ЯA-ZЁ][^.!?]{2,90})$")


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


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
        # склейка непустых нормализованных строк + карта смещений: позиция → номер строки
        parts: list[str] = []
        self._line_starts: list[int] = []
        self._line_ids: list[int] = []
        off = 0
        for i, nl in enumerate(self._norm_lines, start=1):
            if not nl:
                continue
            self._line_starts.append(off)
            self._line_ids.append(i)
            parts.append(nl)
            off += len(nl) + 1
        self._norm_full = " ".join(parts)
        self._char_counts: list[Counter] | None = None   # лениво: для верхней оценки похожести
        self._quote_cache: dict[str, tuple[int | None, float]] = {}

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
        if q in self._quote_cache:
            return self._quote_cache[q]
        res = self._locate(q)
        self._quote_cache[q] = res
        return res

    def _locate(self, q: str) -> tuple[int | None, float]:
        # 1) точное вхождение в строку
        for i, nl in enumerate(self._norm_lines, start=1):
            if q in nl:
                return i, 1.0
        # 2) цитата через границу строк: строка, в которой цитата начинается
        pos = self._norm_full.find(q)
        if pos >= 0:
            k = bisect.bisect_right(self._line_starts, pos) - 1
            return self._line_ids[max(k, 0)], 0.98
        # 3) нечёткое сравнение (LLM слегка перефразировал).
        # Строки перебираются по убыванию точной верхней оценки похожести (пересечение мультимножеств
        # символов — это и есть quick_ratio, а он ≥ ratio); как только оценка ≤ лучшего результата,
        # перебор прекращается. Результат совпадает с полным перебором, но обычно в десятки раз быстрее.
        if self._char_counts is None:
            self._char_counts = [Counter(nl) for nl in self._norm_lines]
        qc = Counter(q)
        lq = len(q)
        bounds: list[tuple[float, int]] = []
        for idx, nl in enumerate(self._norm_lines):
            if not nl:
                continue
            lc = self._char_counts[idx]
            inter = sum(min(n, lc[ch]) for ch, n in qc.items())
            if len(nl) >= lq:
                ub = inter / lq                                   # окно длины |q|: 2·|q∩w|/(2|q|)
            else:
                ub = 2 * inter / (lq + len(nl)) * (len(nl) / lq) ** 0.5
            bounds.append((ub, idx))
        bounds.sort(key=lambda x: (-x[0], x[1]))
        best_i, best = None, 0.0
        for ub, idx in bounds:
            if ub < best:
                break
            nl = self._norm_lines[idx]
            i = idx + 1
            if len(nl) >= lq:
                step = max(1, lq // 4)
                sm = SequenceMatcher(None)
                sm.set_seq2(q)                                    # кэш по цитате (seq2) — переиспользуется
                for start in range(0, len(nl) - lq + 1, step):
                    sm.set_seq1(nl[start:start + lq])
                    if sm.real_quick_ratio() < best or sm.quick_ratio() < best:
                        continue
                    r = _ratio(q, nl[start:start + lq])
                    if r > best or (r == best and best_i is not None and i < best_i):
                        best, best_i = r, i
            else:
                r = _ratio(q, nl) * (len(nl) / lq) ** 0.5
                if r > best or (r == best and best_i is not None and i < best_i):
                    best, best_i = r, i
        return best_i, round(best, 3)

    # --- структура для длинных ТЗ -------------------------------------------
    def outline(self, max_chars: int = 3000) -> str:
        """Оглавление: заголовки с номерами строк (для фрагментов длинного ТЗ)."""
        rows, total = [], 0
        for i, line in enumerate(self.lines, start=1):
            m = _HEADING.match(line)
            if not m:
                continue
            text = (m.group(2) or m.group(4) or "").strip()
            level = len(m.group(1)) if m.group(1) else m.group(3).count(".") + 1
            row = f"{'  ' * (level - 1)}L{i}: {text}"
            total += len(row) + 1
            if total > max_chars:
                rows.append("…")
                break
            rows.append(row)
        return "\n".join(rows)

    def chunks(self, max_chars: int, overlap_lines: int = 2) -> list[tuple[int, int, str]]:
        """Делит ТЗ на фрагменты по разделам (не режет раздел, если он помещается), сохраняя
        исходную нумерацию L<n>. Возвращает [(первая строка, последняя строка, нумерованный текст)]."""
        rows = [(i, f"L{i}: {l.rstrip()}") for i, l in enumerate(self.lines, start=1) if l.strip()]
        if not rows:
            return []
        # границы разделов — заголовки
        sections: list[list[tuple[int, str]]] = []
        for i, row in rows:
            if _HEADING.match(self.lines[i - 1]) or not sections:
                sections.append([])
            sections[-1].append((i, row))
        # слишком большой раздел режем по строкам
        pieces: list[list[tuple[int, str]]] = []
        for sec in sections:
            cur, size = [], 0
            for item in sec:
                if cur and size + len(item[1]) + 1 > max_chars:
                    pieces.append(cur)
                    cur, size = [], 0
                cur.append(item)
                size += len(item[1]) + 1
            if cur:
                pieces.append(cur)
        # жадно упаковываем разделы во фрагменты
        out: list[list[tuple[int, str]]] = []
        cur, size = [], 0
        for piece in pieces:
            psize = sum(len(r) + 1 for _, r in piece)
            if cur and size + psize > max_chars:
                out.append(cur)
                tail = cur[-overlap_lines:] if overlap_lines else []
                cur, size = list(tail), sum(len(r) + 1 for _, r in tail)
            cur.extend(piece)
            size += psize
        if cur:
            out.append(cur)
        return [(c[0][0], c[-1][0], "\n".join(r for _, r in c)) for c in out]

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
                cells, prev_tc = [], None
                for c in row.cells:
                    # объединённая по горизонтали ячейка возвращается несколько раз с тем же _tc;
                    # одинаковый текст в соседних, но разных ячейках сохраняется
                    if prev_tc is not None and c._tc is prev_tc:
                        continue
                    prev_tc = c._tc
                    cells.append(c.text.strip().replace("\n", " "))
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
    """BOM → UTF-32/UTF-16/UTF-8; UTF-16 без BOM — по доле нулевых байтов; затем UTF-8; cp1251 — последним
    (он принимает почти любую последовательность байтов и потому не может стоять раньше)."""
    if data.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return data.decode("utf-32", errors="replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")      # BOM задаёт порядок байтов и отбрасывается
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace")
    # UTF-16 без BOM: старшие байты символов (0x00 — латиница, 0x04 — кириллица и казахские буквы)
    # стоят через один и попадают в диапазон управляющих символов, которых в обычном тексте нет
    head = data[: 4096 - 4096 % 2]
    if len(head) >= 4:
        ctrl = lambda half: sum(1 for b in half if b < 0x09) / max(1, len(half))
        odd, even = ctrl(head[1::2]), ctrl(head[0::2])
        if max(odd, even) > 0.3:
            try:
                return data.decode("utf-16-le" if odd >= even else "utf-16-be")
            except UnicodeDecodeError:
                pass
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1251")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def load_bytes(data: bytes, filename: str) -> Document:
    ext = Path(filename).suffix.lower()
    if ext == ".docx":
        text = _read_docx(data)
    elif ext == ".pdf":
        text = _read_pdf(data)
    else:
        text = _decode(data)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ").replace("\x00", "")
    return Document(name=Path(filename).name, text=text)


def load_path(path: str | Path) -> Document:
    p = Path(path)
    return load_bytes(p.read_bytes(), p.name)


def load_text(text: str, name: str = "pasted.md") -> Document:
    return Document(name=name, text=text.replace("\r\n", "\n"))
