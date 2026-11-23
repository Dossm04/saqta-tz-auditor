"""Детерминированные детекторы: антипаттерны, размытость, численные и смысловые противоречия,
отрицания в контролях. Формулировки в тестах намеренно НЕ совпадают с текстами samples/ —
проверяются обобщённые языковые шаблоны, а не подгонка под демо-ТЗ."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import rules  # noqa: E402
from agents.orchestrator import analyze  # noqa: E402
from config import LLMConfig  # noqa: E402
from knowledge.antipatterns import ANTIPATTERNS  # noqa: E402
from knowledge.domains import OUT_OF_SCOPE, detect_domains  # noqa: E402
from knowledge.fintech_controls import CONTROLS_BY_ID  # noqa: E402
from knowledge.language import CONFLICT_PAIRS  # noqa: E402
from models import Category, Severity  # noqa: E402
from parsing import load_path, load_text  # noqa: E402


def doc(*lines: str):
    return load_text("\n".join(lines))


def ap_ids(*lines: str) -> set[str]:
    return {f.control_id for f in rules.scan_antipatterns(doc(*lines))}


# ── Нормализация строк ────────────────────────────────────────────────────────

@pytest.mark.parametrize("line,expected", [
    ("FR-01. Клиент входит в приложение", "Клиент входит в приложение"),
    ("- NFR-12. Доступность 99.9%", "Доступность 99.9%"),
    ("3.2.1 Переводы по QR", "Переводы по QR"),
    ("1) Отправка уведомлений", "Отправка уведомлений"),
    ("5 минут неактивности — выход из сессии", "5 минут неактивности — выход из сессии"),
    ("2.5 сек — максимальное время ответа", "2.5 сек — максимальное время ответа"),
])
def test_strip_ids_keeps_leading_quantities(line, expected):
    assert rules.strip_ids(line) == expected


def test_heading_detection_does_not_swallow_short_requirements():
    assert rules._is_heading("## 4. Нефункциональные требования")
    assert rules._is_heading("3.1 Регистрация и онбординг")
    assert not rules._is_heading("1. Система должна работать быстро.")
    assert not rules._is_heading("2. Приложение работает быстро")


# ── Размытые формулировки ────────────────────────────────────────────────────

def vague_lines(*lines: str) -> set[int]:
    return {n for f in rules.scan_vague(doc(*lines)) for n in f.lines}


def test_short_numbered_requirement_is_scanned():
    assert vague_lines("1. Выписка формируется быстро.") == {1}


def test_vague_terms_and_terms_of_art():
    lines = [
        "Начисление бонусов выполняется в разумный срок после покупки.",          # 1 размыто
        "Ставка определяется как годовая эффективная ставка вознаграждения.",     # 2 термин
        "В отчёте мерчанта показывается эффективная ставка комиссии за месяц.",   # 3 термин
        "Обработка платежей должна быть эффективной.",                            # 4 размыто
        "Пентест не должен выявить уязвимостей критического и высокого уровня.",  # 5 класс CVSS
        "Комиссии партнёров будут определены после пилота.",                      # 6 TBD (мн. число)
        "Моментальные выплаты в рамки проекта не входят.",                        # 7 вне скоупа
    ]
    assert vague_lines(*lines) == {1, 4, 6}


# ── Численные противоречия ───────────────────────────────────────────────────

def metric_titles(*lines: str) -> list[str]:
    return [f.title for f in rules.scan_metric_conflicts(doc(*lines), semantic=False)]


def test_sla_100_percent_is_not_read_as_zero():
    assert metric_titles("Доступность API — 100% в рабочие часы.", "Доступность личного кабинета: 100%.") == []
    assert any("Доступность" in t for t in metric_titles("Uptime сервиса 99.95%.", "SLA доступности — 99.5% в месяц."))


def test_leading_quantity_session_timeout_conflict():
    titles = metric_titles("15 минут бездействия — сессия закрывается.", "Сессия завершается после 30 минут неактивности.")
    assert any("Тайм-аут сессии" in t for t in titles)


def test_new_metrics_loan_balance_crediting_refund():
    assert any("Сумма займа" in t for t in metric_titles(
        "Сумма кредита: от 50 000 тенге до 2 000 000 тенге.",
        "Максимальная сумма кредита составляет 1 500 000 тенге."))
    # минимальная граница «от 50 000» не участвует в сравнении
    assert metric_titles("Сумма кредита: от 50 000 тенге до 2 000 000 тенге.", "Сумма кредита не превышает 2 000 000 тенге.") == []
    assert any("Остаток" in t for t in metric_titles(
        "Остаток на счёте неидентифицированного клиента не превышает 100 000 ₸.",
        "Для неидентифицированных клиентов максимальный остаток — 150 000 тенге."))
    assert any("Срок зачисления" in t for t in metric_titles(
        "Межбанковский перевод зачисляется получателю не позднее 2 часов.",
        "Средства по межбанковским переводам зачисляются в течение 30 минут."))
    assert any("Срок возврата" in t for t in metric_titles(
        "Покупатель вправе оформить возврат в течение 14 дней.", "Срок на возврат товара — 30 дней с даты покупки."))


def test_different_identification_tiers_are_not_compared():
    assert metric_titles("Для упрощённой идентификации лимит на операцию — 50 000 ₸.",
                         "Для полной идентификации лимит на операцию — 500 000 ₸.") == []
    assert metric_titles("Клиенты с упрощённой идентификацией: суточный лимит — 100 000 ₸.",
                         "Суточный лимит для клиентов с упрощённой идентификацией — 200 000 ₸.") != []


# ── Смысловые противоречия ───────────────────────────────────────────────────

def conflicts(*lines: str) -> dict[str, list[int]]:
    return {f.control_id: f.lines for f in rules.scan_semantic_conflicts(doc(*lines))}


def test_conflict_free_vs_fee_same_topic():
    assert conflicts("Выпуск виртуальной карты бесплатный.", "Комиссия за выпуск карты — 1 000 тенге.") == {"CP-FEE": [1, 2]}
    # разные сегменты (свои счета vs другие банки) — не противоречие
    assert conflicts("Переводы между своими счетами бесплатны.", "Комиссия за переводы в другие банки — 0,5%.") == {}
    # разные темы — не противоречие
    assert conflicts("Пополнение кошелька бесплатное.", "Комиссия за снятие наличных — 1%.") == {}


def test_conflict_confirmation_exception_vs_rule():
    assert conflicts("Платежи на сумму до 5 000 тенге проводятся без ввода OTP.",
                     "Все платежи подтверждаются одноразовым кодом.") == {"CP-CONFIRM": [1, 2]}
    assert conflicts("Ни один платёж не проводится без подтверждения клиентом.", "Все платежи подтверждаются кодом.") == {}
    assert conflicts("Просмотр выписки доступен без подтверждения.",
                     "Каждая расходная операция подтверждается push-уведомлением.") == {}
    # общее правило с явной оговоркой — согласовано
    assert conflicts("Платежи до 5 000 тенге проводятся без подтверждения.",
                     "Все платежи подтверждаются кодом, кроме платежей до 5 000 тенге.") == {}


def test_conflict_net_vs_gross_and_tiers_and_pd():
    assert "CP-NET-GROSS" in conflicts(
        "Комиссия эквайера удерживается при перечислении средств партнёру.",
        "Перечисление партнёру производится в полной сумме, комиссия оплачивается отдельно по счёту.")
    assert "CP-TIER" in conflicts(
        "Пополнение наличными разрешено только полностью идентифицированным клиентам.",
        "При упрощённой идентификации доступно пополнение наличными до 20 000 ₸.")
    assert conflicts("Переводы доступны только полностью идентифицированным клиентам.",
                     "При упрощённой идентификации доступно пополнение до 20 000 ₸.") == {}
    assert "CP-PD-LOCAL" in conflicts(
        "Персональные данные клиентов хранятся на территории Республики Казахстан.",
        "События приложения с номером телефона пользователя отправляются в Mixpanel.")
    assert conflicts("Персональные данные клиентов хранятся на территории Республики Казахстан.",
                     "В Mixpanel передаются только обезличенные события без персональных данных.") == {}


def test_semantic_conflicts_reach_consistency_agent_via_metric_scan():
    d = doc("Выпуск виртуальной карты бесплатный.", "Комиссия за выпуск карты — 1 000 тенге.")
    fs = rules.scan_metric_conflicts(d)
    assert [f.category for f in fs] == [Category.contradiction] and fs[0].control_id == "CP-FEE"


# ── Опасные требования: новые формулировки ───────────────────────────────────

@pytest.mark.parametrize("line,ap", [
    ("Новый пароль высылается пользователю по электронной почте.", "AP-PWD-PLAIN"),
    ("Видеозапись селфи хранится бессрочно.", "AP-BIO-PHOTO"),
    ("Снимки лица клиента архивируются в хранилище банка.", "AP-BIO-PHOTO"),
    ("Код доступа к приложению — четырёхзначный PIN.", "AP-PIN-SHORT"),
    ("Приложение получает доступ к адресной книге и истории вызовов пользователя.", "AP-DEVICE-HARVEST"),
    ("Система ежедневно получает кредитную историю всех клиентов из ГКБ.", "AP-BUREAU-NO-CONSENT"),
    ("Выплата страхового возмещения возможна на счёт третьего лица по заявлению клиента.", "AP-THIRD-PARTY-PAYOUT"),
    ("Информация о просрочке направляется работодателю должника.", "AP-DEBT-DISCLOSURE"),
    ("Полная стоимость кредита показывается клиенту после акцепта оферты.", "AP-GESV-AFTER"),
    ("Партнёр может отправлять номер карты и CVV со своего сервера напрямую в API.", "AP-H2H-CARD"),
    ("В журнал пишется полное тело ответа процессинга.", "AP-FULL-LOG"),
    ("Для постоянных покупателей 3DS не запрашивается.", "AP-3DS-OFF"),
    ("Резервные копии с анкетами клиентов хранятся в Microsoft Azure (Нидерланды).", "AP-PD-ABROAD"),
    ("Изменение номера телефона подтверждается одноразовым кодом, отправленным на новый номер.", "AP-PHONE-CHANGE"),
    ("Реквизиты карты кэшируются в памяти телефона.", "AP-CARD-LOCAL"),
    ("Платежи по QR до 2 000 тенге выполняются без подтверждения.", "AP-NO-CONFIRM"),
])
def test_antipatterns_fire_on_fresh_phrasings(line, ap):
    assert ap in ap_ids(line)


@pytest.mark.parametrize("line", [
    "Фото документа хранится 30 дней, затем удаляется.",
    "С согласия клиента при подаче заявки запрашивается отчёт из ПКБ.",
    "Клиент может перевести деньги на карту другого лица по номеру телефона.",
    "ГЭСВ отображается до подписания договора и повторно после подписания.",
    "В журнал пишутся метаданные запроса без тела и заголовков.",
    "3-D Secure не отключается ни для каких мерчантов.",
    "В Sentry передаются только обезличенные ошибки без персональных данных.",
    "Смена номера подтверждается кодом на прежний номер и селфи.",
    "Возвраты свыше 1 млн тенге без подтверждения руководителя не проводятся.",
    "Просмотр остатка по карте доступен без подтверждения.",
    "Ни одно списание не выполняется без подтверждения клиентом.",
])
def test_antipatterns_respect_negation_and_safeguards(line):
    assert ap_ids(line) == set()


def test_short_pin_requires_absent_attempt_policy():
    assert "AP-PIN-SHORT" in ap_ids("Вход по ПИН-коду из 4 цифр.")
    assert "AP-PIN-SHORT" not in ap_ids("Вход по ПИН-коду из 4 цифр.", "После 3 неверных попыток вход блокируется.")
    assert "AP-PIN-SHORT" not in ap_ids("Вход по ПИН-коду из 4 цифр.", "При вводе неверного ПИН доступ блокируется на час.")


def test_out_of_scope_lines_are_not_requirements():
    assert ap_ids("Не входит в MVP: хранение CVV для повторных платежей.") == set()
    assert OUT_OF_SCOPE.search("Мгновенные переводы не входят в рамки проекта.")


def test_all_antipatterns_have_meeting_metadata():
    for ap in ANTIPATTERNS:
        assert 2 <= len(ap.answer_options) <= 3, ap.id
        assert all(o.strip() for o in ap.answer_options), ap.id
        assert ap.addressee and ap.default_assumption, ap.id
    for cp in CONFLICT_PAIRS:
        assert 2 <= len(cp.answer_options) <= 3 and cp.addressee and cp.default_assumption, cp.id


# ── Отрицание в контролях ─────────────────────────────────────────────────────

def status(cid: str, *lines: str) -> str:
    return rules.check_control(doc(*lines), CONTROLS_BY_ID[cid]).status


def test_check_control_ignores_negated_hits():
    assert status("SEC-PCI-02", "Пополнение с привязанной карты проходит без 3-D Secure.") == "missing"
    assert status("SEC-PCI-02", "Для премиальных клиентов 3DS отключается.") == "missing"
    assert status("SEC-PCI-02", "Все платежи картой проходят 3-D Secure 2.x.") == "covered"
    assert status("SEC-PCI-02", "3-D Secure не отключается ни для каких мерчантов.") == "covered"
    assert status("SEC-PCI-02", "Не входит в MVP: 3-D Secure для платежей.") == "missing"
    assert status("SEC-AUTH-01", "Для входа в кабинет 2FA не требуется.") == "missing"
    line = "двухфакторная аутентификация при входе не применяется"
    assert rules.is_negated(line, 0, len("двухфактор"))
    assert not rules.is_negated("пароль не менее 12 символов", 0, 6)


# ── Регрессия на образцах ─────────────────────────────────────────────────────

def test_wallet_tiers_not_reported_as_operation_limit_conflict():
    r = analyze(load_path(ROOT / "samples" / "wallet_tz.md"), LLMConfig(), use_llm=False)
    titles = [f.title for f in r.findings if f.category == Category.contradiction]
    assert not any("Лимит (операция" in t for t in titles)
    assert any("Лимит (сутки; полная идентификация)" in t for t in titles)


def test_good_tz_stays_clean():
    r = analyze(load_path(ROOT / "samples" / "good_tz.md"), LLMConfig(), use_llm=False)
    assert r.overall_score >= 90
    assert not [f for f in r.findings if f.severity == Severity.critical]


def test_lending_out_of_scope_does_not_create_transfers_domain():
    text = (ROOT / "samples" / "lending_tz.md").read_text(encoding="utf-8")
    assert "transfers" not in detect_domains(text)
