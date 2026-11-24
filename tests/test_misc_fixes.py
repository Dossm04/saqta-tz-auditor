"""Регрессионные тесты исправлений: рекомендуемый вариант вопроса (backfill/kb_meta), повторные ref в
исправлении вопросов, оценка «пустого» ТЗ, сильные стороны со ссылками на реальные строки, нефинтех-ТЗ,
язык ТЗ, скорость поиска дублей, элементы управления содержимым .docx, управляющие символы."""
from __future__ import annotations

import asyncio
import io
import random
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import rules  # noqa: E402
from agents.base import Context  # noqa: E402
from agents.orchestrator import analyze  # noqa: E402
from agents.questions import _APS, _SECTIONS, QuestionAgent, backfill, build_question, kb_meta  # noqa: E402
from agents.scorer import INSUFFICIENT_VERDICT, ScorerAgent, compute_scores, insufficient_text  # noqa: E402
from config import LLMConfig  # noqa: E402
from knowledge.domains import detect_domains, is_fintech  # noqa: E402
from knowledge.fintech_controls import CONTROLS, CONTROLS_BY_ID  # noqa: E402
from models import DocumentModel, Finding, Question, Requirement  # noqa: E402
from parsing import Document, detect_language, load_bytes, load_path, load_text  # noqa: E402

SAMPLES = ROOT / "samples"


def _f(**kw) -> Finding:
    kw.setdefault("agent", "fintech_security")
    kw.setdefault("title", "Находка")
    kw.setdefault("description", "Описание")
    kw.setdefault("category", "security")
    kw.setdefault("severity", "major")
    return Finding(**kw)


# ── #12 / #13: рекомендуемый вариант ─────────────────────────────────────────

def test_kb_meta_first_option_is_recommended_for_every_kb_entry():
    objs = list(CONTROLS_BY_ID.values()) + list(_SECTIONS.values()) + list(_APS.values())
    assert objs
    for obj in objs:
        kb = kb_meta(_f(control_id=obj.id))
        if kb.options:
            assert kb.recommended == 0, obj.id
        else:
            assert kb.recommended is None, obj.id


def test_build_question_stars_first_kb_option_even_if_default_resembles_another():
    # SEC-AUTH-03: эвристика по сходству с допущением выбирала вариант 1; соглашение базы знаний — вариант 0
    q = build_question(_f(id="F-01", title="Не описано: Тайм-аут сессии", control_id="SEC-AUTH-03"))
    assert q.answer_options and q.recommended_option == 0
    q = build_question(_f(id="F-02", title="Не описано: Локализация ПДн", category="compliance",
                          severity="critical", control_id="PRIV-02"))
    assert q.answer_options and q.recommended_option == 0          # раньше None: в отчёте не было звезды


def test_backfill_resets_llm_recommendation_when_kb_options_replace_answers():
    f = _f(id="F-01", title="Не описано: MFA", severity="critical", control_id="SEC-AUTH-01")
    q = Question(question="Нужна ли MFA?", related_findings=["F-01"], answer_options=["Да", "Нет"],
                 recommended_option=1, priority="P0")
    q = backfill(q, {"F-01": f})
    assert q.answer_options == kb_meta(f).options
    assert q.recommended_option == 0                                  # не «1» от LLM из старого списка


def test_backfill_resets_llm_recommendation_when_contradiction_options_replace_answers():
    doc = load_text("# ТЗ\nЛимит перевода — 500 000 ₸ в сутки.\nТекст.\nЛимит перевода — 300 000 ₸ в сутки.\n")
    f = _f(id="F-01", agent="consistency", title="Противоречивые значения: Лимит перевода",
           category="contradiction", lines=[2, 4], evidence=[doc.line(2), doc.line(4)])
    q = Question(question="Какой лимит верный?", related_findings=["F-01"], answer_options=["Уточнить"],
                 recommended_option=0, priority="P1")
    q = backfill(q, {"F-01": f}, doc)
    assert len(q.answer_options) >= 2
    assert q.recommended_option is None                               # у противоречия нет «правильной» стороны


# ── #34: повторные ref в ответе на исправление ───────────────────────────────

class _DupRefLLM:
    def __init__(self):
        self.calls = 0

    async def chat_json(self, system, user, schema):
        self.calls += 1
        if self.calls == 1:
            data = {"questions": [{"question": "Уточните требования к журналам", "priority": "P1",
                                   "related_findings": ["F-02"], "answer_options": ["Да"]}]}
        else:
            good = {"ref": 1, "question": "Какие поля карты допустимо писать в журнал разбора инцидентов?",
                    "priority": "P1", "addressee": "ИБ", "related_findings": ["F-02"],
                    "answer_options": ["Только маскированный PAN (6+4) и ID клиента", "Только токен карты без PAN"],
                    "recommended_option": 0, "default_assumption": "Маскированный PAN и ID клиента."}
            data = {"questions": [dict(good), dict(good), dict(good)]}
        return schema.model_validate(data), 10


def test_repair_counts_each_failing_question_once():
    doc = load_text("# ТЗ\nКлиент переводит деньги.\nЖурнал хранит данные карты.\n")
    ctx = Context(doc=doc, cfg=LLMConfig(api_key="x"))
    ctx.llm = _DupRefLLM()
    ctx.findings = [_f(id="F-02", title="Не полностью описано: Журнал аудита", lines=[3], evidence=[doc.line(3)])]
    asyncio.run(QuestionAgent().run_llm(ctx))
    qq = ctx.report_extra["question_quality"]
    assert ctx.llm.calls == 2
    assert qq.repaired == 1                                           # было 3 — по числу дублей ref
    assert any("журнал" in q.question for q in ctx.questions)


# ── #17 / #24: Scorer ────────────────────────────────────────────────────────

def test_garbage_text_scores_below_defective_sample_with_insufficient_verdict():
    junk = analyze(load_text("a"), LLMConfig(), use_llm=False)
    bad = analyze(load_path(SAMPLES / "sample_tz.md"), LLMConfig(), use_llm=False)
    assert junk.overall_score <= 15 and junk.verdict == INSUFFICIENT_VERDICT and junk.verdict_level == "not_ready"
    dims = {d.key: d.score for d in junk.dimensions}
    assert dims["clarity"] < 100 and dims["consistency"] < 100       # нет текста — нет «идеальной ясности»
    assert bad.overall_score > junk.overall_score
    assert bad.verdict != INSUFFICIENT_VERDICT


def test_insufficient_text_thresholds():
    many = "\n".join(f"- Требование {i}: система должна делать операцию {i}." for i in range(12))
    ctx = Context(doc=load_text(many), cfg=LLMConfig())
    assert insufficient_text(ctx)                                     # строк хватает, требований нет
    ctx.doc_model = DocumentModel(requirements=[Requirement(id=f"R-{i}", text="x", line=i + 1) for i in range(12)])
    assert not insufficient_text(ctx)
    dims, overall = compute_scores(ctx)
    assert {d.key: d.score for d in dims}["clarity"] == 100 and overall > 15
    ctx.doc = load_text("Одна строка.\nВторая строка.\n")
    assert insufficient_text(ctx)


def test_rule_strengths_ignore_requirement_lines_outside_document():
    text = "\n".join(["# ТЗ"] + [f"- FR-{i}. Система должна выполнять операцию {i}." for i in range(1, 13)])
    ctx = Context(doc=load_text(text), cfg=LLMConfig())
    ctx.doc_model = DocumentModel(requirements=[Requirement(id=f"R-{i}", text="x", line=700 + i) for i in range(12)])
    assert not any("Выделено" in s for s in ScorerAgent._strengths(ctx))
    reqs = [Requirement(id=f"R-{i}", text="x", line=i) for i in range(2, 15)]   # L2..L14, а в ТЗ 13 строк
    ctx.doc_model = DocumentModel(requirements=reqs)
    s = next(s for s in ScorerAgent._strengths(ctx) if "Выделено" in s)
    assert "Выделено 12 требований (L2–L13)" in s


# ── #19: ТЗ не про финтех ────────────────────────────────────────────────────

LIBRARY_TZ = ("# ТЗ: сайт школьной библиотеки\n## 1. Назначение\nСайт показывает каталог книг.\n## 2. Требования\n"
              "- Ученик ищет книгу по названию.\n- Библиотекарь добавляет книги через админку.\n"
              "- Сайт должен быстро работать.\n")
SHORT_BANK_TZ = "# ТЗ: перевод денег по номеру телефона\n- Клиент банка отправляет деньги другу.\n- Нужен вход по ПИН.\n"


def test_non_fintech_tz_gets_only_generic_controls():
    doc = load_text(LIBRARY_TZ)
    assert detect_domains(doc.text) == [] and not is_fintech(doc.text)
    checks = {c.control_id: c for c in rules.scan_controls(doc, [])}
    for cid in ("SEC-AUTH-01", "SEC-DATA-01", "REG-SEC-01", "SEC-3RD-01"):
        assert checks[cid].status == "not_applicable" and "финтех" in checks[cid].comment
    for c in CONTROLS:
        if rules.is_generic_control(c):
            assert checks[c.id].status != "not_applicable", c.id
    assert all("*" in CONTROLS_BY_ID[cid].applies_to for cid in rules.GENERIC_CONTROL_IDS)
    r = analyze(doc, LLMConfig(), use_llm=False)
    ids = {f.control_id for f in r.findings}
    assert not ids & {"SEC-AUTH-01", "SEC-DATA-01", "REG-SEC-01"}
    assert not any("pinning" in q.question.lower() for q in r.questions)


def test_short_bank_tz_without_domains_keeps_bank_controls():
    doc = load_text(SHORT_BANK_TZ)
    assert detect_domains(doc.text) == [] and is_fintech(doc.text)
    checks = {c.control_id: c for c in rules.scan_controls(doc, [])}
    assert checks["SEC-AUTH-01"].status != "not_applicable"
    assert checks["REG-SEC-01"].status != "not_applicable"


# ── #18: язык ТЗ ─────────────────────────────────────────────────────────────

EN_TZ = ("# Mobile wallet: technical specification\n## 1. Goals\nThe service lets clients transfer money by phone "
         "number.\n## 2. Functional requirements\nFR-1. The user logs in with a password.\nFR-2. The user can "
         "transfer money to another client.\nFR-3. Card numbers are stored in the database.\n"
         "FR-4. The client can link a second card in the mobile app.\n"
         "## 3. Acceptance criteria\nAll payments are confirmed within 5 seconds.\n")
KK_TZ = ("# Мобильді әмиян: техникалық тапсырма\n## 1. Мақсаты\nКлиенттер телефон нөмірі бойынша аударым жасайды.\n"
         "## 2. Функционалдық талаптар\nFR-1. Пайдаланушы құпиясөзбен кіреді.\nFR-2. Карта нөмірлері "
         "дерекқорда сақталады.\nFR-3. Аударымдар бірден орындалады, төлем расталады.\n"
         "## 3. Қабылдау критерийлері\nБарлық төлемдер 5 секунд ішінде расталады.\n")


def test_detect_language():
    assert detect_language(EN_TZ) == "en" and load_text(EN_TZ).language == "en"
    assert detect_language(KK_TZ) == "kk" and load_text(KK_TZ).language == "kk"
    ru = load_path(SAMPLES / "sample_tz.md")
    assert ru.language == "ru"
    # русское ТЗ с английскими терминами и единичными казахскими словами остаётся русским
    mixed = ru.text + "\nПартнёр: «Қазақстан Халық Банкі», интеграция по REST API (JSON, OAuth 2.0), PCI DSS.\n"
    assert detect_language(mixed) == "ru"
    assert detect_language("ok") == "ru"                              # слишком мало букв — по умолчанию русский


def test_missing_section_findings_are_minor_and_low_confidence_for_foreign_tz():
    for text in (EN_TZ, KK_TZ):
        doc = load_text(text)
        fs = [f for f in rules.section_findings(doc, rules.scan_sections(doc)) if f.title.startswith("Отсутствует")]
        assert fs
        assert all(f.severity.value == "minor" and f.confidence < 0.7 for f in fs)
        assert all("языке" in f.description for f in fs)
        r = analyze(doc, LLMConfig(), use_llm=False)
        assert not [f for f in r.findings if f.title.startswith("Отсутствует раздел") and f.severity.value == "critical"]
    ru = load_text("# ТЗ\nКлиент переводит деньги.\n")
    fs = [f for f in rules.section_findings(ru, rules.scan_sections(ru)) if f.title.startswith("Отсутствует")]
    assert any(f.severity.value == "critical" for f in fs)            # русский ТЗ — без изменений


def test_domains_detect_english_and_kazakh_synonyms():
    assert {"transfers", "cards"} <= set(detect_domains(EN_TZ))
    assert "transfers" in detect_domains(KK_TZ)


# ── #20: поиск дублей ────────────────────────────────────────────────────────

def _brute_duplicates(doc: Document) -> list[tuple[int, int]]:
    reqs = [(i, l.strip()) for i, l in enumerate(doc.lines, start=1) if len(l.strip()) > 50]
    out = []
    for a in range(len(reqs)):
        for b in range(a + 1, min(len(reqs), a + 400)):
            if SequenceMatcher(None, reqs[a][1].lower(), reqs[b][1].lower()).ratio() >= 0.9:
                out.append((reqs[a][0], reqs[b][0]))
    return out[:10]


def test_scan_duplicates_matches_brute_force():
    rnd = random.Random(20)
    base = "Система должна отправлять клиенту уведомление о статусе перевода в течение 5 секунд после операции"
    alph = "абвгдежзийклмнопрстуфхцчшщыэюя 0123456789"
    for _ in range(25):
        lines = []
        for _ in range(45):
            s = list(base if rnd.random() < 0.7 else "".join(rnd.choice(alph) for _ in range(rnd.randint(45, 140))))
            for _ in range(rnd.randint(0, 14)):
                p = rnd.randrange(len(s))
                op = rnd.random()
                if op < 0.4:
                    s[p] = rnd.choice(alph)
                elif op < 0.7:
                    s.insert(p, rnd.choice(alph))
                elif len(s) > 52:
                    s.pop(p)
            if rnd.random() < 0.1:
                s = list("абвг" * rnd.randint(13, 30))              # вырожденные повторы
            lines.append("".join(s))
        doc = Document(name="x", text="\n".join(lines))
        assert [tuple(f.lines) for f in rules.scan_duplicates(doc)] == _brute_duplicates(doc)


def test_scan_duplicates_is_fast_on_long_tz():
    rnd = random.Random(7)
    words = load_path(SAMPLES / "wallet_tz.md").text.split()
    lines = [f"FR-{i}. " + " ".join(rnd.choice(words) for _ in range(rnd.randint(9, 30))) + "." for i in range(1500)]
    lines.insert(105, lines[100].replace(".", ";", 1))
    doc = Document(name="long", text="\n".join(lines))
    t = time.perf_counter()
    found = rules.scan_duplicates(doc)
    elapsed = time.perf_counter() - t
    assert any(f.lines == [101, 106] for f in found)
    assert elapsed < 5.0, elapsed                                      # было ~25 с


# ── #29: элементы управления содержимым в .docx ──────────────────────────────

def _sdt_docx() -> bytes:
    import docx
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    d = docx.Document()
    d.add_paragraph("Перед контентконтролем")
    body = d.element.body

    def sdt_with(*children):
        sdt, content = OxmlElement("w:sdt"), OxmlElement("w:sdtContent")
        sdt.append(OxmlElement("w:sdtPr"))
        for ch in children:
            content.append(ch)
        sdt.append(content)
        return sdt

    def para(text):
        p, r, t = OxmlElement("w:p"), OxmlElement("w:r"), OxmlElement("w:t")
        t.text = text
        r.append(t)
        p.append(r)
        return p

    tmp = docx.Document().add_table(rows=1, cols=2)
    tmp.cell(0, 0).text, tmp.cell(0, 1).text = "Лимит", "500 000 ₸"
    nested = sdt_with(para("Вложенный раздел в sdt"))
    block = sdt_with(para("Абзац внутри sdt"), tmp._tbl, nested)
    body.insert(len(body) - 1, block)                                 # перед sectPr
    # строчный sdt внутри абзаца (поле формы)
    p = d.add_paragraph("Ответственный: ")
    inline = OxmlElement("w:sdt")
    content = OxmlElement("w:sdtContent")
    r, t = OxmlElement("w:r"), OxmlElement("w:t")
    t.text = "Иванов И. И."
    r.append(t)
    content.append(r)
    inline.append(OxmlElement("w:sdtPr"))
    inline.append(content)
    p._p.append(inline)
    assert qn("w:sdt")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_docx_content_controls_are_read():
    doc = load_bytes(_sdt_docx(), "t.docx")
    text = doc.text
    assert "Перед контентконтролем" in text
    assert "Абзац внутри sdt" in text and "Вложенный раздел в sdt" in text
    assert "| Лимит | 500 000 ₸ |" in text
    assert "Ответственный: Иванов И. И." in text


# ── #30: управляющие символы ─────────────────────────────────────────────────

def test_control_characters_are_stripped_and_docx_export_works():
    from report_docx import to_docx

    raw = "# ТЗ\x02 кошелёк\nПереводы\x03 по номеру телефона выполняются мгновенно.\x1a\nСтраница 1\x0cСтраница 2\n"
    for doc in (load_bytes(raw.encode("utf-8"), "t.txt"), load_text(raw)):
        assert not any(ord(ch) < 32 and ch not in "\n\t" for ch in doc.text)
        assert doc.lines[2] == "Страница 1" and doc.lines[3] == "Страница 2"   # \f — граница строки, как и прежде
    r = analyze(load_bytes(raw.encode("utf-8"), "t.txt"), LLMConfig(), use_llm=False)
    assert to_docx(r)                                                 # раньше: ValueError (XML compatible)
    assert load_text("a\r\nb\rc").lines == ["a", "b", "c"]
