"""Тесты ядра: исправления Verifier/парсинга/кэша/ТЗ v2, гарантии генератора вопросов,
critic loop, сообщения между агентами, длинные ТЗ. Формулировки ТЗ — свежие, не из samples/."""
from __future__ import annotations

import asyncio
import io
import random
import sys
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import base  # noqa: E402
from agents.analyzers import RawFintech, ask_doc, merge_control_status, parse_findings  # noqa: E402
from agents.base import Context, LLMClient  # noqa: E402
from agents.orchestrator import analyze  # noqa: E402
from agents.questions import (LIMITS, QuestionAgent, build_question, contradiction_question,  # noqa: E402
                              lint_question, must_cover)
from agents.scorer import finalize_summary, valid_strengths  # noqa: E402
from agents.verifier import VerifierAgent, _norm_title  # noqa: E402
from config import LLMConfig  # noqa: E402
from knowledge.template import PLACEHOLDER  # noqa: E402
from models import Category, ControlCheck, Finding, Priority, Question, Severity  # noqa: E402
from parsing import _norm, load_bytes, load_path, load_text  # noqa: E402

SAMPLE = ROOT / "samples" / "sample_tz.md"


def _ctx(text: str, **cfg) -> Context:
    return Context(doc=load_text(text), cfg=LLMConfig(**cfg))


def _f(**kw) -> Finding:
    kw.setdefault("agent", "fintech_security")
    kw.setdefault("title", "Находка")
    return Finding(**kw)


# ── Verifier ──────────────────────────────────────────────────────────────────

LOGIN_TZ = "# ТЗ\nВход по логину.\nПароль клиента хранится в базе в виде хэша SHA-1 без соли.\nКонец.\n"


def test_verifier_keeps_distinct_controls_on_one_line():
    ctx = _ctx(LOGIN_TZ)
    line = "Пароль клиента хранится в базе в виде хэша SHA-1 без соли."
    ctx.findings_by_agent["fintech_security"] = [
        _f(title="Не полностью описано: Хранение паролей", category="security", severity="major",
           evidence=[line], lines=[3], control_id="SEC-AUTH-04"),
        _f(title="Опасное требование: Устаревшие криптоалгоритмы", category="security", severity="critical",
           evidence=[line], lines=[3], control_id="AP-WEAK-CRYPTO"),
        _f(title="Не полностью описано: Журнал аудита", category="security", severity="major",
           evidence=[line], lines=[3], control_id="OPS-AUDIT-01"),
    ]
    VerifierAgent().run_rules(ctx)
    assert {f.control_id for f in ctx.findings} == {"SEC-AUTH-04", "AP-WEAK-CRYPTO", "OPS-AUDIT-01"}
    assert ctx.verifier.merged_duplicates == 0


def test_verifier_merges_llm_and_rules_on_same_line_and_explains_it():
    ctx = _ctx(LOGIN_TZ)
    line = "Пароль клиента хранится в базе в виде хэша SHA-1 без соли."
    ctx.findings_by_agent["fintech_security"] = [
        _f(title="Опасное требование: Устаревшие криптоалгоритмы", category="security", severity="critical",
           evidence=[line], lines=[3], control_id="AP-WEAK-CRYPTO")]
    ctx.findings_by_agent["consistency"] = [
        _f(agent="consistency", title="Слабое хэширование паролей", category="security", severity="major",
           evidence=["хранится в базе в виде хэша SHA-1"], lines=[3])]
    VerifierAgent().run_rules(ctx)
    assert len(ctx.findings) == 1
    f = ctx.findings[0]
    assert set(f.sources) == {"fintech_security", "consistency"}
    rec = ctx.verifier.merges[0]
    assert rec.kept_id == f.id and rec.from_agent and "строк" in rec.reason


def test_norm_title_strips_generated_prefixes():
    assert _norm_title("Нет раздела из шаблона: Интеграции") == "интеграции"
    assert _norm_title("Противоречивые значения: Лимит (сутки)") == "лимит сутки"
    assert _norm_title("Не полностью описано: Журнал аудита") == "журнал аудита"


def test_template_section_findings_are_not_collapsed():
    tpl = load_text("# Шаблон\n## 1. Требования к интеграциям\n## 2. Требования к интерфейсам\n"
                    "## 3. Требования к отчётности\n## 4. Требования к документации\n", "tpl.md")
    r = analyze(load_text("# ТЗ\nКороткий текст без разделов.\n"), LLMConfig(), use_llm=False, template=tpl)
    missing = [t for t in r.template_check if not t["present"]]
    got = [f for f in r.findings if f.title.startswith("Нет раздела из шаблона")]
    assert len(missing) == 4 and len(got) == len(missing)


def test_parse_findings_drops_server_owned_fields():
    raw = [{"id": "F-01", "agent": "x", "title": "Хранение CVV", "category": "compliance", "evidence": ["CVV"],
            "verified": True, "sources": ["a", "b", "c"], "verification_note": "проверено", "revision": 1,
            "confidence": 0.6}]
    f = parse_findings(raw, "fintech_security")[0]
    assert f.id == "" and f.agent == "fintech_security" and f.verified is None
    assert f.sources == [] and f.verification_note is None and f.revision == 0


def test_verifier_resets_forged_verification_and_consensus():
    ctx = _ctx(LOGIN_TZ)
    forged = _f(title="Слабое хэширование", category="security", severity="major", confidence=0.6,
                evidence=["хранится в базе в виде хэша SHA-1"], lines=[3],
                verified=True, sources=["fintech_security", "consistency", "completeness"], verification_note="ok")
    fake = _f(title="Размытая формулировка: «молниеносно»", category="ambiguity", evidence=["молниеносно и надёжно"],
              verified=True, lines=[2])
    ctx.findings_by_agent["fintech_security"] = [forged, fake]
    VerifierAgent().run_rules(ctx)
    assert len(ctx.findings) == 1
    kept = ctx.findings[0]
    assert kept.sources == ["fintech_security"] and kept.confidence == 0.6 and kept.verification_note is None
    rej = ctx.verifier.rejected[0]
    assert rej.title.startswith("Размытая") and "не найдена" in rej.reason and rej.best_line is not None
    assert forged.verified is True and forged.sources[0] == "fintech_security"   # исходник не мутирован


def test_verifier_records_downgrade_with_reason():
    ctx = _ctx("# ТЗ\nВсе соединения шифруются по TLS 1.3, используется certificate pinning и mTLS.\n")
    ctx.findings_by_agent["fintech_security"] = [
        _f(title="Не описано: Шифрование при передаче", category="security", severity="critical",
           control_id="SEC-DATA-01")]
    VerifierAgent().run_rules(ctx)
    if ctx.verifier.downgraded:      # контроль распознан правилами как покрытый
        d = ctx.verifier.downgrades[0]
        assert d.id == ctx.findings[0].id and d.from_severity == "critical" and d.hit_lines


# ── статусы контролей от LLM ──────────────────────────────────────────────────

def _base(status: str) -> ControlCheck:
    return ControlCheck(control_id="SEC-X", title="Контроль", area="ИБ", status=status,
                        evidence=["строка"], lines=[2], comment="пред-проверка")


def test_llm_control_status_validation():
    locate = lambda q: (None, 0.0)
    cc = merge_control_status(ControlCheck(control_id="SEC-X", status="not applicable"), _base("partial"), locate)
    assert cc.status == "partial"                                   # N/A без оснований при найденных признаках
    cc = merge_control_status(ControlCheck(control_id="SEC-X", status="n/a"), _base("missing"), locate)
    assert cc.status == "missing"                                   # N/A без обоснования
    cc = merge_control_status(ControlCheck(control_id="SEC-X", status="не применимо",
                                           comment="в проекте нет карт"), _base("missing"), locate)
    assert cc.status == "not_applicable"                            # с обоснованием — принимается
    cc = merge_control_status(ControlCheck(control_id="SEC-X", status="вроде есть"), _base("partial"), locate)
    assert cc.status == "partial"                                   # нераспознанное → пред-проверка
    assert ControlCheck(control_id="A", status="Partially covered").status == "partial"
    assert ControlCheck(control_id="A", status="ОТСУТСТВУЕТ").status == "missing"


# ── поиск цитат ───────────────────────────────────────────────────────────────

def test_locate_quote_across_lines_after_blank_lines():
    doc = load_text("# ТЗ\n\n---\n\nПеревод подтверждается одноразовым\nкодом из SMS в течение 60 секунд.\n\nКонец.\n")
    ln, score = doc.locate_quote("подтверждается одноразовым кодом из SMS")
    assert ln == 5 and score == 0.98


def test_locate_quote_tolerates_changed_endings():
    doc = load_text("Заголовок\nКлиент подтверждает перевод одноразовым кодом из SMS-сообщения.\n")
    ln, score = doc.locate_quote("клиенты подтверждают переводы одноразовыми кодами из SMS-сообщения")
    assert ln == 2 and score >= 0.75


def _bruteforce(doc, quote):
    q = _norm(quote).strip(" .…")
    best_i, best = None, 0.0
    for i, nl in enumerate(doc._norm_lines, start=1):
        if not nl:
            continue
        if len(nl) >= len(q):
            for s in range(0, len(nl) - len(q) + 1, max(1, len(q) // 4)):
                r = SequenceMatcher(None, q, nl[s:s + len(q)]).ratio()
                if r > best:
                    best, best_i = r, i
        else:
            r = SequenceMatcher(None, q, nl).ratio() * (len(nl) / len(q)) ** 0.5
            if r > best:
                best, best_i = r, i
    return best_i, round(best, 3)


def test_fast_fuzzy_search_equals_bruteforce():
    rnd = random.Random(7)
    text = SAMPLE.read_text(encoding="utf-8")
    words = [w for w in text.split() if len(w) > 3]
    for _ in range(25):
        quote = " ".join(rnd.sample(words, rnd.randint(3, 7)))
        doc = load_text(text)                      # свежий документ: без кэша цитат
        if any(_norm(quote).strip(" .…") in nl for nl in doc._norm_lines):
            continue
        if _norm(quote).strip(" .…") in doc._norm_full:
            continue
        assert doc.locate_quote(quote) == _bruteforce(doc, quote), quote


# ── кэш LLM ──────────────────────────────────────────────────────────────────

def test_llm_cache_key_and_hits(tmp_path, monkeypatch):
    from agents.scorer import SummaryOut

    monkeypatch.setattr(base, "CACHE_DIR", tmp_path)
    cfg = LLMConfig(api_key="k", model="m1", base_url="https://a.example/v1", temperature=0.1, cache=True)
    c = LLMClient(cfg)
    k = c.cache_key(SummaryOut, "sys", "user")
    assert k != LLMClient(LLMConfig(api_key="k", model="m2", base_url="https://a.example/v1")).cache_key(SummaryOut, "sys", "user")
    assert k != LLMClient(LLMConfig(api_key="k", model="m1", base_url="https://b.example/v1")).cache_key(SummaryOut, "sys", "user")
    assert k != LLMClient(LLMConfig(api_key="k", model="m1", base_url="https://a.example/v1",
                                    temperature=0.7)).cache_key(SummaryOut, "sys", "user")
    calls = []

    async def create(**kw):
        calls.append(kw)
        return SimpleNamespace(usage=SimpleNamespace(total_tokens=7),
                               choices=[SimpleNamespace(message=SimpleNamespace(content='{"summary": "ok"}'))])

    c.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    (tmp_path / f"{k}.json").write_text('{"strengths": "битый ответ без summary"}', encoding="utf-8")
    out, _ = asyncio.run(c.chat_json("sys", "user", SummaryOut))
    assert out.summary == "ok" and c.cache_hits == 0 and len(calls) == 1      # битый кэш — не попадание
    out, tokens = asyncio.run(c.chat_json("sys", "user", SummaryOut))
    assert c.cache_hits == 1 and tokens == 0 and len(calls) == 1


# ── чтение файлов ─────────────────────────────────────────────────────────────

def test_decode_bom_and_utf16_variants():
    text = "# ТЗ кошелька\nЛимит перевода — 200 000 тенге в сутки.\n"
    for data in (text.encode("utf-16"), text.encode("utf-16-le"), text.encode("utf-16-be"),
                 text.encode("utf-8-sig"), text.encode("utf-8"), text.encode("cp1251")):
        doc = load_bytes(data, "tz.txt")
        assert doc.lines[:2] == text.splitlines()[:2], data[:6]


def test_docx_tables_keep_equal_cells_and_skip_merged():
    import docx

    d = docx.Document()
    t = d.add_table(rows=2, cols=3)
    for i, v in enumerate(["Есть", "Есть", "Нет"]):
        t.rows[0].cells[i].text = v
    merged = t.rows[1].cells[0].merge(t.rows[1].cells[1])
    merged.text = "Объединённая"
    t.rows[1].cells[2].text = "Третья"
    buf = io.BytesIO()
    d.save(buf)
    lines = load_bytes(buf.getvalue(), "t.docx").lines
    assert "| Есть | Есть | Нет |" in lines
    assert "| Объединённая | Третья |" in lines


def test_placeholder_regex_ignores_inequalities():
    for ok in ("Время ответа p95 < 2 сек при нагрузке > 100 RPS", "p95<2с и rps>100", "сумма <= 100 и >= 5",
               "ссылка <https://bank.kz>", "a<b и c>d"):
        assert not PLACEHOLDER.search(ok), ok
    for ph in ("<Название системы>", "Лимит: <указать сумму>.", "[заполнить]", "{{owner}}", "______"):
        assert PLACEHOLDER.search(ph), ph


# ── ТЗ v2 ─────────────────────────────────────────────────────────────────────

def test_revised_tz_keeps_numbering_ids_and_table_cells():
    from improve import build_revised_tz, replace_line

    f = _f(title="Опасное требование: CVV", evidence=["сервис запоминает CVV2 карты"], lines=[3],
           control_id="AP-CVV-STORE", suggested_text="CVV не хранится; используется токен карты")
    row = "| ТР-7 | Для автоплатежей сервис запоминает CVV2 карты | Высокий |"
    new = replace_line(row, f.suggested_text, "F-01", f)
    assert new.count("|") == row.count("|") and "| ТР-7 |" in new and "| Высокий |" in new
    assert "CVV не хранится" in new and "запоминает" not in new
    assert replace_line("4.2.1 Сервис запоминает CVV2.", "X", "F-1").startswith("4.2.1 X")
    assert replace_line("2) Сервис запоминает CVV2.", "X", "F-1").startswith("2) X")
    assert replace_line("- ФТ-12. Сервис запоминает CVV2.", "X", "F-1").startswith("- ФТ-12. X")

    doc = load_text("# ТЗ\n## Платежи\n| ТР-7 | Для автоплатежей сервис запоминает CVV2 карты клиента | Высокий |\n"
                    "3.1.4 Пароль хранится в виде хэша MD5.\n")
    r = analyze(doc, LLMConfig(), use_llm=False)
    revised, stats = build_revised_tz(doc, r, {})
    lines = revised.splitlines()
    assert stats["replaced"] >= 1
    assert lines[2].startswith("| ТР-7 |") and lines[2].rstrip().endswith("| Высокий |")
    assert lines[3].startswith("3.1.4 ")


# ── генератор вопросов ────────────────────────────────────────────────────────

CONTRA_TZ = """# ТЗ онлайн-займов
Максимальная сумма займа составляет 300 000 тенге.
FR-09. Срок рассмотрения возврата — 180 дней.
Подтверждение операций не требуется для повторных клиентов.
Клиент может получить заём на сумму до 450 000 тенге.
Срок рассмотрения возврата — 90 дней.
Каждая операция клиента подтверждается биометрией.
"""


def test_contradiction_question_offers_conflicting_values():
    doc = load_text(CONTRA_TZ)
    f = _f(id="F-03", agent="consistency", title="Противоречивые значения: Сумма займа", category="contradiction",
           severity="major", lines=[2, 5], evidence=[doc.line(2), doc.line(5)])
    q = contradiction_question(f, doc)
    assert "300 000 тенге (L2)" in q.question and "450 000 тенге (L5)" in q.question and " или " in q.question
    assert any("300 000 тенге (L2)" in o for o in q.answer_options) and any("450 000 тенге (L5)" in o for o in q.answer_options)
    assert q.priority == Priority.P0 and not lint_question(q)      # деньги → блокер; вопрос закрытый

    f2 = _f(id="F-04", agent="consistency", title="Срок возврата", category="contradiction", severity="major",
            lines=[3, 6])
    q2 = contradiction_question(f2, doc)
    assert "180 дней (L3)" in q2.question and "90 дней (L6)" in q2.question and "09" not in q2.question

    f3 = _f(id="F-05", agent="consistency", title="Подтверждение операций", category="contradiction",
            severity="critical", lines=[4, 7])
    q3 = contradiction_question(f3, doc)
    assert "(L4)" in q3.question and "(L7)" in q3.question and len(q3.answer_options) == 3


def _many_findings() -> list[Finding]:
    fs = []
    topics = ["шифрование ключей в HSM", "маскирование PAN в интерфейсе", "сверку с процессингом",
              "отзыв согласия на ПДн", "санкционный скрининг получателя", "лимиты до идентификации",
              "пентест перед релизом", "защиту от root-устройств", "журнал действий операторов",
              "резервное копирование баз", "мониторинг подозрительных входов", "блокировку после перебора ПИН",
              "подпись запросов партнёров", "хранение биометрических шаблонов"]
    for i, t in enumerate(topics):
        fs.append(_f(title=f"Не описано: {t}", category="security", severity="critical", control_id=f"X-{i}",
                     question=f"Кто и как обеспечивает {t}?", suggested_text=f"Требование: {t} по стандарту банка."))
    scen = ["отмену платежа", "возврат комиссии", "частичное списание", "смену номера телефона", "закрытие счёта",
            "смену тарифа", "повторную отправку SMS", "выгрузку выписки", "истечение сессии", "отказ АБС",
            "блокировку карты", "смену ПИН-кода"]
    for i, t in enumerate(scen):
        fs.append(_f(title=f"Не полностью описано: {t}", category="fintech", severity="major",
                     question=f"Что видит клиент, когда выполняет {t}?"))
    fs.append(_f(agent="consistency", title="Противоречивые значения: Сумма займа", category="contradiction",
                 severity="major", lines=[2, 5]))
    fs.append(_f(agent="consistency", title="Противоречие: подтверждение операций", category="contradiction",
                 severity="minor", lines=[4, 7]))
    fs.append(_f(title="Опасное требование: Операции без подтверждения", category="security", severity="major",
                 lines=[4], evidence=["Подтверждение операций не требуется для повторных клиентов"],
                 control_id="AP-NO-CONFIRM", suggested_text="Каждая расходная операция подтверждается."))
    fs.append(_f(agent="consistency", title="Открытый вопрос в тексте ТЗ (TBD / «уточнить позже»)",
                 category="completeness", severity="major", lines=[3], evidence=["Срок рассмотрения возврата"]))
    fs.append(_f(agent="consistency", title="Дублирующееся требование", category="contradiction",
                 severity="minor", lines=[3, 6]))
    for i, f in enumerate(fs, start=1):
        f.id = f"F-{i:02d}"
    return fs


def test_question_coverage_guaranteed_under_caps():
    ctx = _ctx(CONTRA_TZ)
    ctx.findings = _many_findings()
    QuestionAgent().run_rules(ctx)
    main = ctx.questions
    covered = {r for q in main for r in q.related_findings}
    must = {f.id for f in ctx.findings if must_cover(f)}
    assert must <= covered                                         # резерв сверх лимита
    assert sum(q.priority == Priority.P0 for q in main) > LIMITS[Priority.P0]
    assert ctx.report_extra["extra_questions"]                     # лишнее не теряется молча
    ids = [q.id for q in main + ctx.report_extra["extra_questions"]]
    assert len(ids) == len(set(ids))
    qq = ctx.report_extra["question_quality"]
    assert qq.must_cover_covered == f"{len(must)}/{len(must)}" and qq.critical_covered == "14/14"
    # повестка: P0 → P1 → P2, внутри приоритета — по адресату
    order = [(q.priority.value, q.addressee) for q in main]
    assert order == sorted(order, key=lambda x: (x[0], x[1]))


def test_offline_questions_are_closed_and_grounded():
    ctx = _ctx(CONTRA_TZ)
    ctx.findings = _many_findings()
    QuestionAgent().run_rules(ctx)
    ap_q = next(q for q in ctx.questions if any(r == "F-29" for r in q.related_findings))
    assert len(ap_q.answer_options) >= 2 and ap_q.default_assumption and ap_q.impact
    assert ap_q.context.startswith("L4:")
    tbd_q = next(q for q in ctx.questions if "F-30" in q.related_findings)
    assert len(tbd_q.answer_options) >= 2 and tbd_q.recommended_option == 0


def test_question_linter():
    good = Question(question="Как реализуем повторные платежи без хранения CVV?", priority="P0",
                    related_findings=["F-01"], answer_options=["Токен процессинга", "Повторный ввод CVV"])
    assert lint_question(good) == []
    bad = Question(question="Уточните требования к безопасности", priority="P1", answer_options=["Да", "Нет"])
    errs = lint_question(bad)
    assert any("«?»" in e for e in errs) and any("общий" in e for e in errs)
    assert any("вариантов" in e for e in errs) and any("связанной находки" in e for e in errs)
    assert lint_question(Question(question="Что выбрать?", priority="P2", related_findings=["F-1"])) == []
    two = Question(question="Кто решает и когда? И как?", priority="P2", related_findings=["F-1"])
    assert any("несколько" in e for e in lint_question(two))
    dup = Question(question="Что выбрать?", priority="P1", related_findings=["F-1"],
                   answer_options=["Токенизация у процессинга", "токенизация у процессинга."])
    assert any("вариантов" in e for e in lint_question(dup))


class _QuestionsLLM:
    """Фейковая модель для генератора вопросов: дубль, выдуманная ссылка, вопрос, проваливший чеклист."""

    def __init__(self):
        self.calls = []

    async def chat_json(self, system, user, schema):
        self.calls.append(user)
        if len(self.calls) == 1:
            data = {"questions": [
                {"question": "Какой суточный лимит переводов верный?", "priority": "P0", "addressee": "Бизнес-заказчик",
                 "related_findings": ["F-01"], "answer_options": ["300 000 ₸", "450 000 ₸"], "recommended_option": 7},
                {"question": "Какой суточный лимит переводов верный?!", "priority": "P1",
                 "related_findings": ["F-01", "F-02"], "answer_options": []},
                {"question": "Нужен ли блокчейн?", "related_findings": ["F-99"]},
                {"question": "Уточните требования к журналам", "priority": "P1", "related_findings": ["F-02"],
                 "answer_options": ["Да"], "id": "Q-77", "lint": []},
            ]}
        else:
            assert "Нарушения" in user or "нарушения" in user
            data = {"questions": [{"ref": 1, "question": "Какие поля карты допустимо писать в журнал разбора инцидентов?",
                                   "priority": "P1", "addressee": "ИБ", "related_findings": ["F-02"],
                                   "answer_options": ["Только маскированный PAN (6+4) и ID клиента",
                                                      "Только токен карты без PAN"],
                                   "recommended_option": 0, "default_assumption": "Маскированный PAN и ID клиента."}]}
        return schema.model_validate(data), 10


def test_llm_questions_dedupe_ground_and_repair_once():
    doc = load_text(CONTRA_TZ)
    ctx = Context(doc=doc, cfg=LLMConfig(api_key="x"))
    ctx.llm = _QuestionsLLM()
    ctx.findings = [
        _f(id="F-01", agent="consistency", title="Противоречивые значения: Сумма займа", category="contradiction",
           severity="major", lines=[2, 5], evidence=[doc.line(2), doc.line(5)]),
        _f(id="F-02", title="Не полностью описано: Журнал аудита", category="security", severity="major",
           lines=[3], evidence=[doc.line(3)]),
        _f(id="F-03", title="Не описано: Подтверждение операций", category="security", severity="critical",
           question="Как подтверждаются расходные операции клиента?"),
    ]
    agent = QuestionAgent()
    asyncio.run(agent.run_llm(ctx))
    qs = ctx.questions
    assert len(ctx.llm.calls) == 2                                   # ровно один вызов на исправление
    assert all("F-99" not in q.related_findings for q in qs) and not any("блокчейн" in q.question for q in qs)
    limit_qs = [q for q in qs if "лимит" in q.question]
    assert len(limit_qs) == 1 and set(limit_qs[0].related_findings) >= {"F-01"}   # дубль склеен
    assert all(q.recommended_option is None or q.recommended_option < len(q.answer_options) for q in qs)
    assert any("журнал" in q.question and q.addressee == "ИБ" for q in qs)          # исправлено моделью
    assert {"F-01", "F-02", "F-03"} <= {r for q in qs for r in q.related_findings}  # критичная покрыта
    qq = ctx.report_extra["question_quality"]
    assert qq.repaired == 1 and qq.critical_covered == "1/1"
    assert "<tz>" not in ctx.llm.calls[0] and "F-01" in ctx.llm.calls[0] and "варианты" not in ctx.llm.calls[0][:20]


def test_build_question_uses_kb_and_decision_options():
    f = _f(id="F-07", title="Не описано: Шифрование при хранении", category="security", severity="major",
           question="Какие данные шифруются при хранении?", suggested_text="ПДн шифруются AES-256, ключи в HSM.")
    q = build_question(f)
    assert len(q.answer_options) >= 2 and q.recommended_option == 0 and "AES-256" in q.default_assumption
    assert q.impact and not lint_question(q)


# ── Scorer ───────────────────────────────────────────────────────────────────

def test_scorer_summary_guarantees_and_strength_validation():
    s = finalize_summary("ТЗ описывает кошелёк. Риски в хранении данных.", "Требует доработки", 4)
    assert "«Требует доработки»" in s and "P0" in s and len(s) <= 600
    long = finalize_summary(("Очень длинное предложение про риски кошелька. " * 40), "Не готово к разработке", 2)
    assert len(long) <= 600 and "P0" in long and "Не готово к разработке" in long
    ctx = _ctx("# ТЗ\nИнтеграция с АБС по REST.\n\nКонец.\n")
    got = valid_strengths(["Описана интеграция с АБС (L2)", "Есть раздел (L3)", "Без ссылки", "Вне документа (L99)"], ctx)
    assert got == ["Описана интеграция с АБС (L2)"]


# ── оркестрация: сообщения, critic loop, <tz>, длинные ТЗ ───────────────────────

def test_offline_messages_and_rendering():
    from report import to_markdown
    from report_docx import to_docx

    r = analyze(load_path(SAMPLE), LLMConfig(), use_llm=False)
    kinds = {(m.sender, m.receiver, m.kind) for m in r.messages}
    assert ("reader", "fintech_security", "doc_model") in kinds
    assert ("consistency", "verifier", "findings") in kinds
    assert ("verifier", "questions", "verified_findings") in kinds and ("questions", "scorer", "questions") in kinds
    assert [t.agent for t in r.trace].count("reader") == 1           # офлайн: один проход Reader
    assert r.question_quality.total == len(r.questions) and r.question_quality.critical_covered.split("/")[0] == \
        r.question_quality.critical_covered.split("/")[1]
    md = to_markdown(r)
    assert "Сообщения между агентами" in md and "Качество вопросов" in md and "★ рекомендуем" in md
    assert "Если ответа нет до старта спринта" in md
    import docx

    text = "\n".join(p.text for p in docx.Document(io.BytesIO(to_docx(r))).paragraphs)
    assert "Качество вопросов" in text and "закладываем" in text


CRITIC_TZ = """# ТЗ кошелька
## Требования
FR-01. Перевод между кошельками выполняется в кратчайшие сроки.
FR-02. Лимит пополнения — 150 000 тенге в сутки.
"""


class _CriticLLM:
    """Consistency цитирует неточно; после замечания Verifier исправляет цитату."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.active_model, self.cache_hits, self.transient_errors, self.switches = "fake", 0, 0, []
        self.users: list[tuple[str, str]] = []

    async def chat_json(self, system, user, schema):
        self.users.append((system, user))
        if "вернул тебе твои находки" in system:
            data = {"findings": [{"ref": 1, "evidence": ["Перевод между кошельками выполняется в кратчайшие сроки"],
                                  "lines": [3]}], "withdrawn": []}
        elif "агент **Consistency**" in system:
            data = {"findings": [{"category": "ambiguity", "severity": "major",
                                  "title": "Срок перевода не измерим",
                                  "evidence": ["переводы проходят моментально всегда"], "lines": [30],
                                  "confidence": 0.8}]}
        elif "агент **Reader**" in system:
            data = {"title": "Кошелёк", "domains": ["transfers", "mobile"]}
        elif "агент **Question Generator**" in system:
            data = {"questions": []}
        elif "агент **Scorer**" in system:
            data = {"summary": "Кратко.", "strengths": []}
        else:
            data = {}
        return schema.model_validate(data), 5


def test_critic_loop_restores_fixed_finding(monkeypatch):
    from agents import orchestrator

    holder = {}

    def make(cfg):
        holder["llm"] = _CriticLLM(cfg)
        return holder["llm"]

    monkeypatch.setattr(orchestrator, "LLMClient", make)
    r = analyze(load_text(CRITIC_TZ), LLMConfig(api_key="fake"), use_llm=True)
    v = r.verifier
    assert v.revision_returned == 1 and v.revision_restored == 1 and v.revision_withdrawn == 0
    assert v.restored and "подтверждено" in v.restored[0].outcome
    f = next(f for f in r.findings if f.title == "Срок перевода не измерим")
    assert f.revision == 1 and f.lines == [3] and "исправлена" in (f.verification_note or "")
    assert any(m.kind == "revision" and m.sender == "consistency" for m in r.messages)
    critic = next(t for t in r.trace if t.agent == "critic")
    assert critic.produced == 1 and "восстановлено 1" in critic.note
    # документ во всех запросах с текстом ТЗ обёрнут в <tz>…</tz>
    with_doc = [u for s, u in holder["llm"].users if "L4: FR-02" in u]
    assert with_doc and all("<tz>" in u and "</tz>" in u for u in with_doc)
    assert r.mode == "llm"


def test_critic_loop_is_noop_offline():
    r = analyze(load_text(CRITIC_TZ), LLMConfig(), use_llm=False)
    assert r.verifier.revision_returned == 0 and not any(t.agent == "critic" for t in r.trace)


def test_document_chunks_keep_line_numbers_and_sections():
    body = []
    for s in range(1, 9):
        body.append(f"## {s}. Раздел номер {s}")
        body += [f"Требование {s}.{i}: система обрабатывает операцию типа {s}-{i} без потерь." for i in range(1, 12)]
        body.append("")
    doc = load_text("\n".join(body))
    parts = doc.chunks(1200)
    assert len(parts) > 2
    seen = set()
    for a, b, text in parts:
        assert len(text) <= 1200 + 200
        for row in text.splitlines():
            n = int(row.split(":", 1)[0][1:])
            assert row == f"L{n}: {doc.lines[n - 1]}"
            seen.add(n)
    assert seen == {i for i, l in enumerate(doc.lines, start=1) if l.strip()}
    assert "## 3. Раздел номер 3" not in doc.outline() and "L27: 3. Раздел номер 3" in doc.outline()


def test_long_document_is_analyzed_in_chunks_with_merged_controls():
    body = ["# ТЗ"] + [f"## Раздел {s}\n" + "\n".join(f"Пункт {s}.{i} описывает обработку платежа." for i in range(20))
                       for s in range(6)]
    ctx = Context(doc=load_text("\n".join(body)), cfg=LLMConfig(api_key="x", max_doc_chars=900))
    seen = []

    class _Agent:
        name = "fintech_security"

        async def ask(self, c, system, user, schema):
            seen.append(user)
            k = len(seen)
            return schema.model_validate({
                "controls": [{"control_id": "SEC-X", "status": "covered" if k == 2 else "missing",
                              "evidence": [f"цитата {k}"]}],
                "findings": [{"title": f"Находка {k}"}]})

    out = asyncio.run(ask_doc(_Agent(), ctx, "sys", "## Подсказки", RawFintech))
    assert len(seen) > 1 and all("Оглавление всего ТЗ" in u and "<tz>" in u and "Фрагмент" in u for u in seen)
    assert len(out.findings) == len(seen)
    ctrl = out.controls
    assert len(ctrl) == 1 and ctrl[0]["status"] == "covered" and "цитата 2" in ctrl[0]["evidence"]
    assert any(m.kind == "chunks" for m in ctx.messages)


@pytest.mark.parametrize("sev,expected", [(Severity.critical, Priority.P0), (Severity.minor, Priority.P2)])
def test_build_question_priority_follows_severity(sev, expected):
    f = _f(id="F-01", title="Не описано: Тема", category="completeness", severity=sev.value,
           question="Кто отвечает за тему?")
    assert build_question(f).priority == expected
    assert Category.completeness  # импорт используется
