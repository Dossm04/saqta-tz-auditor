"""Domain Critic: доменные правила находят дефекты и не шумят на корректных формулировках."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import iso_domain  # noqa: E402
from agents.segmenter import segment  # noqa: E402
from parsing import load_path, load_text  # noqa: E402

TPL = """# Тест
### Функциональные требования
* **Ф-1.** {f1}
* **Ф-2.** {f2}
### Нефункциональные требования
{nfr}
### Требования к данным
{data}
### Требования к интеграции
{integ}
"""


def _topics(md: str) -> dict[str, set[str]]:
    doc = load_text(md, "t.md")
    seg = segment(doc)
    out: dict[str, set[str]] = {}
    for u in seg.units:
        for i in iso_domain.unit_rules(seg, doc)[u.rid]:
            out.setdefault(u.doc_ref, set()).add(i.agent.split(":")[1])
    return out


def test_domain_rules_find_typical_fintech_gaps():
    t = _topics(TPL.format(
        f1="Месяц определяется по дате совершения покупки в Алматы. В расходы входят проведённые покупки. Возврат уменьшает расходы месяца.",
        f2="При достижении порога клиент получает push, максимум один в месяц.",
        nfr="Сервис отправляет push в течение минуты после события.",
        data="Хранить идентификатор карты, сумму в тиынах и порог в тенге.",
        integ="Повторная доставка события сохраняет исходный итог."))
    assert {"time", "reversal"} <= t["Ф-1"]
    assert "nfr" in next(v for k, v in t.items() if k.startswith("Нефункциональные"))
    data = next(v for k, v in t.items() if k.startswith("Требования к данным"))
    assert {"money", "retention"} <= data
    assert "idempotency" in next(v for k, v in t.items() if k.startswith("Требования к интеграции"))


def test_domain_rules_quiet_on_well_specified_requirements():
    t = _topics(TPL.format(
        f1="Месяц определяется по времени UTC+5 (Asia/Almaty) по дате проведения операции.",
        f2="Карта закрывается безвозвратно; для неактивной или заблокированной карты настройка скрыта.",
        nfr="95% событий передаются в сервис push не позднее 60 с при нагрузке до 200 событий/с.",
        data="Суммы хранятся в тиынах целым числом, порог вводится в тенге и пересчитывается. Срок хранения — 5 лет, затем удаление.",
        integ="Повторный запрос с тем же идентификатором операции возвращает исходный результат."))
    assert not any(t.values()), t


def test_domain_rules_are_quiet_on_reference_good_document():
    doc = load_path(ROOT / "samples" / "good_tz.md")
    seg = segment(doc)
    assert not any(iso_domain.unit_rules(seg, doc).values())


def test_case_document_acceptance_coverage_gap():
    doc = load_path(ROOT / "examples" / "case_requirements.md")
    issues = iso_domain.set_rules(segment(doc), doc)
    assert issues and "возврат" in issues[0].problem and "1 000" in issues[0].problem
