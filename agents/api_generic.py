"""Проверки спецификации API в произвольном оформлении и сверка числовых ограничений по всему документу.

Шаблон кейса (поля Parameter/Format/Mandatory, «Статусы, код NNN») разбирают правила в iso_review и iso_domain.
Здесь — то же для типового Markdown-оформления: строка «`POST /api/v1/...`», таблица «Параметр | Тип | ...»,
curl-пример, «Пример ответа (200 OK)» с JSON, таблица кодов ответа. Работает по строкам документа, поэтому
не зависит от того, как сегментатор нарезал раздел.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from parsing import Document

_EP = re.compile(r"(?<![\w-])(GET|POST|PUT|PATCH|DELETE)\s+`?(/[\w/{}\-.:]+)")
_HEADING = re.compile(r"^\s*#{1,6}\s+(.*)")
_CURL = re.compile(r"\bcurl\b", re.I)
_TABLE_HEAD = re.compile(r"^\s*\|.*(?:парамет|поле|name|field).*\|.*(?:тип|type|format|формат)", re.I)
_EXAMPLE = re.compile(r"(?:пример|example|ответ|response)[^\n]{0,60}?\(?\b([1-5]\d\d)\b", re.I)
_CODE_ROW = re.compile(r"^\s*\|\s*`?([1-5]\d\d)`?\s*(?:[A-Za-z ]+)?\|")
_CODE_LINE = re.compile(r"^\s*(?:[-*]\s*)?`?([1-5]\d\d)`?(?:\s+[A-Za-z][A-Za-z ]+)?\s*[—–:.-]\s+\S")

_STATE_PATH = re.compile(r"/(?:close|cancel|delete|remove|open|create|confirm|pay|transfer|topup|freeze|unfreeze|block|unblock|"
                         r"activate|deactivate|update|approve|reject|book|submit|terminate|withdraw)\b", re.I)
_STATE_HEAD = re.compile(r"расторж|закрыт|отмен|удален|создан|оплат|перевод|подтвержд|изменен|блокир|заморо|разморо|пополн|"
                         r"открыти|запис[ьи]\b|бронир|выставлен|отключен|подключен", re.I)
_SENSITIVE = re.compile(r"^(?:otp|code|sms_?code|pin|password|pass|pwd|iin|иин|pan|card(?:_?number)?|cvv|cvc|phone|"
                        r"phone_?number|token|access_?token|passport|snils|birth_?date|email)$", re.I)
_MONEY_PATH = re.compile(r"pay|transfer|topup|open|withdraw|charge|refund|debit|оплат|перевод|пополн|списан|открыти|платеж|платёж|"
                         r"автоплат|погашен", re.I)
_IDEM = re.compile(r"идемпотент|idempoten|x-request-id|request-?id|ключ\w*\s+(?:дедупл|идемпот)|дедупликац", re.I)
_OWNER = re.compile(r"принадлеж|владел\w*|owner|из\s+(?:сесси|токен)|соответств\w*\s+(?:клиент|пользоват|сесси|токен)", re.I)
_ERR_BODY = re.compile(r"\"(?:status|result)\"\s*:\s*\"(?:error|fail\w*|declined|rejected)\"|\"error\"\s*:|недостаточно|"
                       r"\"code\"\s*:\s*\"[A-Z_]*(?:ERROR|FAIL|INSUFFICIENT|INVALID|NOT_FOUND|DENIED|EXCEEDED)", re.I)


@dataclass
class Endpoint:
    method: str
    path: str
    line: int                       # 1-based
    heading: str
    params: dict[str, tuple[str, int, str]] = field(default_factory=dict)   # имя → (тип, строка, описание)
    curls: list[tuple[str, int]] = field(default_factory=list)              # (склеенная команда, строка)
    examples: list[tuple[str, int, str]] = field(default_factory=list)      # (код, строка, тело)
    end: int = 0


def endpoints(doc: Document) -> list[Endpoint]:
    lines = doc.lines
    eps: list[Endpoint] = []
    heading = ""
    for i, raw in enumerate(lines):
        h = _HEADING.match(raw)
        if h:
            heading = h.group(1).strip()
            continue
        if _CURL.search(raw) or raw.strip().startswith("-"):
            continue
        m = _EP.search(raw)
        if m and len(raw.strip()) < 140:
            eps.append(Endpoint(m.group(1), m.group(2).rstrip("`"), i + 1, heading))
    for k, ep in enumerate(eps):
        ep.end = eps[k + 1].line - 1 if k + 1 < len(eps) else len(lines)
        # конец блока — следующий заголовок того же или более высокого уровня после первого заголовка эндпоинта
        _fill(ep, lines)
    return eps


def _fill(ep: Endpoint, lines: list[str]) -> None:
    i = ep.line
    header = None
    while i < ep.end:
        raw = lines[i]
        if _TABLE_HEAD.match(raw):
            header = [c.strip().lower() for c in raw.strip().strip("|").split("|")]
        elif header and raw.strip().startswith("|") and not re.match(r"^\s*\|[\s:|-]+\|\s*$", raw):
            cells = [c.strip().strip("`") for c in raw.strip().strip("|").split("|")]
            if len(cells) >= 2:
                ti = next((j for j, c in enumerate(header) if re.search(r"тип|type|format|формат", c)), 1)
                name = cells[0]
                if re.match(r"^[A-Za-z_][\w.]*$", name):
                    ep.params[name] = (cells[ti].lower() if ti < len(cells) else "", i + 1, " ".join(cells[ti + 1:]))
        elif not raw.strip().startswith("|"):
            header = None if raw.strip() else header
        if _CURL.search(raw):
            start, cmd = i + 1, raw
            while cmd.rstrip().endswith("\\") and i + 1 < ep.end:
                i += 1
                cmd = cmd.rstrip()[:-1] + " " + lines[i].strip()
            ep.curls.append((cmd, start))
        m = _EXAMPLE.search(raw)
        if m and not _CURL.search(raw):
            body, j = [], i + 1
            while j < min(ep.end, i + 14) and not _EXAMPLE.search(lines[j]) and not _HEADING.match(lines[j]):
                body.append(lines[j])
                j += 1
            ep.examples.append((m.group(1), i + 1, "\n".join(body)))
        i += 1


def response_codes(doc: Document) -> dict[str, int]:
    """Коды из таблицы/списка кодов ответа: код → строка."""
    out: dict[str, int] = {}
    for i, raw in enumerate(doc.lines):
        m = _CODE_ROW.match(raw) or _CODE_LINE.match(raw)
        if m:
            out.setdefault(m.group(1), i + 1)
    return out


def _query(url: str) -> dict[str, str]:
    q = url.split("?", 1)[1] if "?" in url else ""
    out = {}
    for part in re.split(r"[&\s\"']", q):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def _json_values(text: str) -> dict[str, str]:
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"\"(\w+)\"\s*:\s*(\"[^\"]*\"|[-\d.]+|true|false|null)", text)}


def api_issues(doc: Document, mk) -> list:
    """mk(code, sev, topic, problem, evidence, fix, lines) → Issue."""
    eps = endpoints(doc)
    if not eps:
        return []
    out = []
    text = doc.text
    has_idem = bool(_IDEM.search(text))
    has_owner = bool(_OWNER.search(text))
    L = doc.lines

    def q(line: int, n: int = 90) -> str:
        s = L[line - 1].strip().strip("|`* ").strip()
        return s[:n].rsplit(" ", 1)[0] if len(s) > n else s

    for ep in eps:
        name = f"{ep.method} {ep.path}"
        # 1. изменение состояния методом GET
        if ep.method == "GET" and (_STATE_PATH.search(ep.path) or _STATE_HEAD.search(ep.heading)):
            out.append(mk("set_consistent", "critical", "api",
                          f"Метод {name} ({ep.heading}) изменяет состояние, но объявлен как GET. GET должен быть безопасным и "
                          "идемпотентным: запрос повторят прокси, кэш или предзагрузка браузера, и операция выполнится без ведома клиента.",
                          [q(ep.line)], "Использовать POST (или DELETE/PATCH по смыслу), а GET оставить только для чтения.", [ep.line]))
        # 2. чувствительные данные в query-строке / в URL
        found = {}
        for cmd, ln in ep.curls:
            urls = re.findall(r"https?://\S+", cmd)
            for u in urls:
                for k in _query(u):
                    if _SENSITIVE.match(k):
                        found.setdefault(k, ln)
        for k, (t, ln, desc) in ep.params.items():
            if _SENSITIVE.match(k) and re.search(r"query|в\s+url|строке\s+запроса", desc, re.I):
                found.setdefault(k, ln)
        if found:
            lines = sorted(set(found.values()))
            out.append(mk("security", "critical", "security",
                          f"В {name} чувствительные данные ({', '.join(found)}) передаются в query-строке URL. URL попадает в логи "
                          "веб-сервера и прокси, историю и заголовок Referer — это утечка персональных данных и кодов подтверждения.",
                          [q(lines[0])], "Передавать эти поля только в теле POST-запроса, в логах маскировать.", lines))
        # 3. ошибка с кодом 200
        for code, ln, body in ep.examples:
            if code == "200" and _ERR_BODY.search(body):
                out.append(mk("set_consistent", "major", "api",
                              f"В {name} ошибка возвращается с кодом 200 OK (пример в строке {ln}). Клиент и мониторинг сочтут "
                              "запрос успешным; ошибку можно распознать только разбором тела ответа.",
                              [q(ln)], "Возвращать ошибку с кодом 4xx (например, 409 или 422) и описать её в перечне кодов.", [ln]))
                break
        # 4. тип параметра в таблице не совпадает с примером
        vals: dict[str, tuple[str, int, str]] = {}
        for cmd, ln in ep.curls:
            for u in re.findall(r"https?://\S+", cmd):
                for k, v in _query(u).items():
                    vals.setdefault(k, (v, ln, "query"))
            mbody = re.search(r"(?:-d|--data(?:-raw)?)\s+'(.*?)'", cmd)
            if mbody:
                for k, v in _json_values(mbody.group(1)).items():
                    vals.setdefault(k, (v, ln, "json"))
        for k, (t, pln, desc) in ep.params.items():
            if k not in vals:
                continue
            v, vln, where = vals[k]
            bad = False
            if where == "json":
                if t.startswith(("int", "number", "decimal", "float", "long")) and v.startswith('"'):
                    bad = True
                elif t.startswith("string") and not v.startswith('"') and v not in ("null",):
                    bad = True
                elif t.startswith("bool") and v not in ("true", "false"):
                    bad = True
            else:
                fmt = re.search(r"`([^`]+)`", desc) or re.search(r"формат\w*\s+(\S+)", desc, re.I)
                if t.startswith(("int", "number", "decimal")) and not re.fullmatch(r"-?[\d.]+", v):
                    bad = True
                elif fmt and re.search(r"[A-Za-zА-Яа-я]", fmt.group(1)) and re.fullmatch(r"\d+", v):
                    bad = True
            if bad:
                out.append(mk("set_consistent", "major", "api",
                              f"В {name} параметр {k} описан в таблице как {t or 'другой тип'}, а в примере передаётся {v}: "
                              "разработчики клиента и сервера реализуют разные форматы.",
                              [q(pln), q(vln)], f"Привести тип и формат параметра {k} к одному значению в таблице и примерах.",
                              sorted({pln, vln})))
        # 5. денежная операция без идемпотентности
        if ep.method in ("POST", "PUT", "PATCH") and not has_idem and (_MONEY_PATH.search(ep.path) or _MONEY_PATH.search(ep.heading)) \
                and re.search(r"amount|сумм", " ".join(ep.params) + " " + " ".join(c for c, _ in ep.curls), re.I):
            out.append(mk("set_complete", "critical", "idempotency",
                          f"Денежная операция {name} не защищена от повтора: нет ключа идемпотентности. При таймауте клиент "
                          "повторит запрос, и деньги спишутся дважды.",
                          [q(ep.line)], "Добавить заголовок Idempotency-Key, срок его хранения и ответ на повторный запрос.", [ep.line]))
        # 6. объект по идентификатору без проверки владельца
        if re.search(r"\{\w*(?:id|Id|ID)\}", ep.path) and not has_owner:
            pid = re.search(r"\{(\w*(?:id|Id|ID))\}", ep.path).group(1)
            extra = [k + 1 for k in range(ep.line, ep.end) if pid in L[k] and not L[k].lstrip().startswith("|")
                     and not _CURL.search(L[k])][:3]
            out.append(mk("security", "critical", "security",
                          f"{name}: объект выбирается по идентификатору из URL, но не сказано, что сервис проверяет его принадлежность "
                          "клиенту из сессии. Подставив чужой идентификатор, можно действовать с чужим объектом (IDOR).",
                          [q(ep.line)], "Явно требовать проверку принадлежности объекта клиенту из токена; иначе 403/404.",
                          sorted({ep.line, *extra})))
    # однотипные находки по нескольким методам (IDOR, идемпотентность) — одной записью со всеми строками
    res, first = [], {}
    for i in out:
        kind = "idor" if "IDOR" in i.problem else "idem" if "идемпотентност" in i.problem else None
        if kind and kind in first:
            f = first[kind]
            f.lines = sorted(set(f.lines + i.lines))
            f.evidence = (f.evidence + i.evidence)[:3]
            f.problem = f.problem.replace(" не защищена", " (и другие методы) не защищена", 1) if kind == "idem" else f.problem
            continue
        if kind:
            first[kind] = i
        res.append(i)
    return res


# ── числовые ограничения: одно и то же ограничение с разными значениями ────────────────────────

_LIMIT = re.compile(r"(не более|не больше|не выше|не свыше|максимум|максимальн\w*(?:\s+\w+){0,2}?\s*[—:–-]?|не менее|минимум|"
                    r"минимальн\w*(?:\s+\w+){0,2}?\s*[—:–-]?|до|от)\s+(\d{1,3}(?:[\s  ]\d{3})+|\d+(?:[.,]\d+)?)\s*(млн|млрд|тыс\.?)?\s*"
                    r"(₸|тенге|тг\b|%|[а-яё]{3,}(?:\s+[а-яё]{3,}){0,2})", re.I)
_ADJ = re.compile(r"(?:ых|их|ой|ый|ий|ая|яя|ое|ее|ые|ие|ого|его|ому|ему)$")
_SKIP_NOUN = re.compile(r"^(?:секунд|минут|час|мс|милли|дн|день|дня|рабоч|календ|месяц|лет|год|раз[аи]?$|числ|символ|знак|цифр|"
                        r"строк|элемент|запис|процент|года|недел|сутк|утра|вечера)", re.I)
_SUBJ_STOP = {"сумма", "суммы", "сумму", "должна", "должен", "может", "более", "менее", "клиент", "клиента", "система", "который",
              "которые", "только", "также", "одного", "одной", "каждый", "каждого", "будет", "можно", "нельзя", "больше"}


def _num(s: str, mult: str | None) -> float:
    v = float(re.sub(r"[\s  ]", "", s).replace(",", "."))
    m = (mult or "").lower()
    return v * (1e6 if m.startswith("млн") else 1e9 if m.startswith("млрд") else 1e3 if m.startswith("тыс") else 1)


def limit_conflicts(doc: Document, mk) -> list:
    facts = []
    for i, raw in enumerate(doc.lines):
        low = raw.lower().replace("ё", "е")
        for m in _LIMIT.finditer(low):
            word = m.group(1)
            words = m.group(4).split()
            noun = next((w for w in words if not _ADJ.search(w)), words[0]) if words[0] not in ("₸", "%") else words[0]
            if _SKIP_NOUN.match(noun) or (word in ("до", "от") and noun not in ("₸", "тенге", "тг") and not m.group(3)):
                continue
            if re.match(r"\d{1,2}:\d{2}", low[m.start(2):m.start(2) + 5]):
                continue
            kind = "max" if re.match(r"не более|не больше|не выше|не свыше|максим|до", word) else "min"
            noun_key = "тенге" if noun in ("₸", "тенге", "тг") else noun[:6]
            before = re.findall(r"[а-яё]{5,}", low[max(0, m.start() - 70):m.start()])
            subj = {w[:6] for w in before[-4:]} - {w[:6] for w in _SUBJ_STOP}
            facts.append((kind, noun_key, _num(m.group(2), m.group(3)), i + 1, subj, m.group(0).strip()))
    out, used = [], set()
    for a in range(len(facts)):
        for b in range(a + 1, len(facts)):
            fa, fb = facts[a], facts[b]
            if fa[0] != fb[0] or fa[1] != fb[1] or fa[2] == fb[2] or abs(fa[3] - fb[3]) < 3:
                continue
            same_subject = bool(fa[4] & fb[4]) if fa[1] in ("тенге", "%") else (bool(fa[4] & fb[4]) or len(fa[1]) >= 5)
            if not same_subject or (fa[3], fb[3]) in used or fa[3] in {x for p in used for x in p}:
                continue
            used.add((fa[3], fb[3]))
            word = "максимум" if fa[0] == "max" else "минимум"
            out.append(mk("set_consistent", "critical", "limits",
                          f"Одно ограничение ({word}) задано разными значениями: «{fa[5]}» (строка {fa[3]}) и «{fb[5]}» (строка {fb[3]}). "
                          "Неясно, какое значение реализовать и проверять.",
                          [doc.lines[fa[3] - 1].strip().strip("|*` ")[:90], doc.lines[fb[3] - 1].strip().strip("|*` ")[:90]],
                          "Оставить одно значение в одном месте (например, в разделе бизнес-правил) и ссылаться на него.",
                          [fa[3], fb[3]]))
            if len(out) >= 6:
                return out
    return out
