"""Определение предметных областей ТЗ — от них зависит, какие контроли применимы."""
from __future__ import annotations

import re

DOMAIN_LABELS = {
    "transfers": "Переводы и платежи",
    "cards": "Платёжные карты",
    "kyc": "Онбординг / KYC",
    "lending": "Кредитование",
    "deposits": "Депозиты / счета",
    "mobile": "Мобильное приложение",
    "web": "Веб-кабинет",
    "api": "API / интеграции",
    "biometrics": "Биометрия",
    "crypto": "Цифровые активы",
}

DOMAIN_PATTERNS: dict[str, str] = {
    "transfers": r"перевод|p2p|платеж|оплат|qr|kaspi|счет получател|получател",
    "cards": r"карт[аыуое]|pan\b|cvv|cvc|visa|mastercard|эквайринг|unionpay",
    "kyc": r"kyc|идентификац|верификац|регистрац|онбординг|удостоверени|\bиин\b|паспорт",
    "lending": r"кредит|займ|рассрочк|гэсв|микрофинанс|ипотек",
    "deposits": r"депозит|вклад|текущ\w* счет|открыти\w* счет",
    "mobile": r"мобильн|ios|android|приложени|смартфон",
    "web": r"веб|web|браузер|личн\w* кабинет|интернет-банк",
    "api": r"\bapi\b|интеграц|open ?banking|rest|webhook|шлюз|процессинг",
    "biometrics": r"биометр|liveness|селфи|selfie|распознаван\w* лиц|face ?id|отпечат",
    "crypto": r"крипт|цифров\w* актив|блокчейн|токен\w* (bitcoin|usdt)|stablecoin",
}

# Домен засчитывается, только если есть хотя бы один «сильный» признак: «график платежей»
# или «платёжный терминал» в кредитном ТЗ ещё не делают его ТЗ на переводы.
DOMAIN_STRONG: dict[str, str] = {
    "transfers": r"перевод|p2p|получател|\bqr\b|kaspi|эквайринг|платежн\w* шлюз|оплат\w* (услуг|покуп|товар|по qr)",
}


OUT_OF_SCOPE = re.compile(
    r"не вход(ит|ят)|out of scope|вне (рамок|скоуп)|за рамк\w* (проекта|mvp|релиза|скоупа|работ|тз|данного)|"
    r"следующ\w* (релиз|этап)|релиз\w*\s*[2-9]|в будущ\w* (релиз|верси)|не реализу",
    re.IGNORECASE,
)


def in_scope_lines(text: str) -> list[str]:
    """Строки, которые описывают скоуп проекта (без строк «не входит в MVP / релиз 2»)."""
    return [l for l in text.splitlines() if not OUT_OF_SCOPE.search(l)]


def detect_domains(text: str, min_hits: int = 2) -> list[str]:
    # строки «не входит в MVP / релиз 2» не делают домен применимым
    t = "\n".join(in_scope_lines(text)).lower().replace("ё", "е")
    found = []
    for dom, pat in DOMAIN_PATTERNS.items():
        hits = len(re.findall(pat, t))
        if hits < min_hits:
            continue
        strong = DOMAIN_STRONG.get(dom)
        if strong and not re.search(strong, t):
            continue
        found.append(dom)
    return found
