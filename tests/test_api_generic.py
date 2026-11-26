"""Проверки API в произвольном оформлении, сверка лимитов и правила покрытия (роли, коды, время)."""

from pathlib import Path

from agents import api_generic as A
from agents.iso_domain import _coverage_rules, _mk, set_rules, unit_rules
from agents.segmenter import segment
from parsing import load_path, load_text

ROOT = Path(__file__).resolve().parents[1]

API_DOC = """# Сервис

## Часть A
### Функциональные требования
- **Ф-1.** Клиент может иметь не более 3 открытых счетов.
- **Ф-2.** Заявка после 23:00 обрабатывается следующим днём.

## Часть B. REST API

### Закрытие счёта

`GET /api/v1/accounts/{accountId}/close`

| Параметр | Тип | Обязательный | Описание |
|---|---|---|---|
| accountId | string | да (path) | Идентификатор |

### Оплата

`POST /api/v1/payments/pay`

| Параметр | Тип | Обязательный | Описание |
|---|---|---|---|
| amount | integer | да | Сумма |

```bash
curl -X POST "https://api.example.kz/api/v1/payments/pay?otp=123456&iin=900101300123" -d '{"amount": "100"}'
```

Пример ответа (200 OK):

```json
{"status": "ERROR", "message": "Превышен лимит: не более 5 счетов на клиента"}
```

### Коды ответов

| Код | Значение |
|---|---|
| 200 | Успешно |
| 401 | Не авторизован |
"""


def _mkr(code, sev, topic, problem, evidence, fix, lines):
    return _mk(code, sev, topic, problem, None, evidence, fix, lines=lines)


def test_free_form_api_checks_find_typical_defects():
    doc = load_text(API_DOC, "api.md")
    eps = A.endpoints(doc)
    assert [e.method for e in eps] == ["GET", "POST"]
    text = " ".join(i.problem for i in A.api_issues(doc, _mkr))
    assert "объявлен как GET" in text
    assert "query-строке" in text and "otp" in text
    assert "200 OK" in text
    assert "параметр amount" in text
    assert "идемпотентност" in text
    assert "IDOR" in text


def test_limit_conflict_between_requirement_and_api_message():
    doc = load_text(API_DOC, "api.md")
    issues = A.limit_conflicts(doc, _mkr)
    assert any("не более 3" in i.problem and "не более 5" in i.problem for i in issues)


def test_coverage_rules_codes_and_time_of_day():
    doc = load_text(API_DOC, "api.md")
    text = " ".join(i.problem for i in _coverage_rules(segment(doc), doc))
    assert "404" in text and "403" in text
    assert "23:00" in text


def test_roles_rule():
    doc = load_text("# ТЗ\n## Функциональные требования\n- **Ф-1.** Врач вносит заключение.\n"
                    "- **Ф-2.** Регистратор отменяет запись пациента.\n", "r.md")
    assert any(i.agent == "domain:roles" for i in _coverage_rules(segment(doc), doc))


def test_table_requirement_ids_and_code_tables_are_recognized():
    doc = load_text("# ТЗ\n## Нефункциональные требования\n| № | Требование |\n|---|---|\n| НФ-1 | Ответ за 2 секунды |\n\n"
                    "## Коды ответов\n| Код | Значение |\n|---|---|\n| 404 | Не найдено |\n", "t.md")
    seg = segment(doc)
    refs = {u.doc_ref: u for u in seg.units}
    assert "НФ-1" in refs and refs["НФ-1"].explicit_id
    assert any(r.endswith("код 404") for r in refs)


def test_new_rules_stay_quiet_on_reference_document():
    doc = load_path(ROOT / "samples" / "good_tz.md")
    seg = segment(doc)
    assert set_rules(seg, doc) == []
    assert sum(len(v) for v in unit_rules(seg, doc).values()) == 0
