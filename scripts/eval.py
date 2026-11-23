"""Оценка качества ревьюера на ТЗ с заранее размеченными дефектами (golden labels).

    python scripts/eval.py                 # офлайн-правила по всем eval/*.expected.json
    python scripts/eval.py --llm           # LLM-агенты (нужен LLM_API_KEY)
    python scripts/eval.py --min-recall 0.7 --max-fp 0   # порог для CI

Метрики: полнота (recall) по посаженным дефектам с разбивкой по типам и ложные
срабатывания на строках, которые помечены как «не флагать».
"""
from __future__ import annotations

import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.orchestrator import analyze  # noqa: E402
from config import LLMConfig  # noqa: E402
from knowledge.fintech_controls import CONTROLS_BY_ID  # noqa: E402
from knowledge.tz_structure import SECTIONS  # noqa: E402
from models import Category, Finding, Report  # noqa: E402
from parsing import load_path  # noqa: E402

SEC = {Category.security, Category.compliance, Category.fintech}
SECTION_TITLES = {s.id: s.title for s in SECTIONS}


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _overlap(f: Finding, lines: list[int]) -> int:
    return len(set(f.lines) & set(lines))


def is_detected(p: dict, r: Report) -> bool:
    kind, lines, cid = p["kind"], p.get("lines", []), p.get("control_id")
    fs = r.findings
    if cid and any(f.control_id == cid for f in fs):
        return True
    if kind in ("antipattern", "weakness"):
        return any(f.category in SEC and _overlap(f, lines) for f in fs)
    if kind == "contradiction":
        need = min(2, len(lines))
        return any(f.category == Category.contradiction and _overlap(f, lines) >= need for f in fs)
    if kind in ("vague", "tbd"):
        return any(_overlap(f, lines) and f.category not in SEC for f in fs)
    if kind == "missing_control":
        # пробел засчитывается, если контроль помечен в матрице как отсутствующий/частичный
        # или если есть находка с похожим заголовком (LLM мог не указать control_id)
        if any(c.control_id == cid and c.status in ("missing", "partial") for c in r.controls):
            return True
        title = CONTROLS_BY_ID[cid].title if cid in CONTROLS_BY_ID else p["description"]
        return any(f.category in SEC and not f.evidence and _sim(f.title, title) >= 0.45 for f in fs)
    if kind == "missing_section":
        title = SECTION_TITLES.get(cid, p["description"])
        return any(f.category in (Category.completeness, Category.testability) and not f.evidence
                   and (_sim(f.title, title) >= 0.4 or _sim(f.title, p["description"]) >= 0.4) for f in fs)
    return any(_overlap(f, lines) for f in fs)


def evaluate(expected_file: Path, use_llm: bool, cfg: LLMConfig) -> dict:
    exp = json.loads(expected_file.read_text(encoding="utf-8"))
    stem = expected_file.name.removesuffix(".expected.json")
    if "sample" not in exp:
        cands = [f"samples/{stem}.md", f"samples/{stem}_tz.md"]
        exp["sample"] = next((c for c in cands if (ROOT / c).exists()), cands[-1])
    sample = ROOT / exp["sample"]
    r = analyze(load_path(sample), cfg, use_llm=use_llm)
    by_kind: dict[str, list[int]] = {}
    missed = []
    for p in exp["planted"]:
        ok = is_detected(p, r)
        by_kind.setdefault(p["kind"], [0, 0])
        by_kind[p["kind"]][1] += 1
        if ok:
            by_kind[p["kind"]][0] += 1
        else:
            missed.append(f'{p["id"]} [{p["kind"]}] {p["description"]}')
    fps = []
    for m in exp.get("must_not_flag", []):
        for f in r.findings:
            if f.evidence and _overlap(f, m["lines"]):
                fps.append(f'L{m["lines"]}: {f.id} {f.title} — {m["why"]}')
    detected = sum(v[0] for v in by_kind.values())
    total = sum(v[1] for v in by_kind.values())
    return {"sample": exp["sample"], "domain": exp.get("domain", ""), "detected": detected, "total": total,
            "recall": detected / total if total else 1.0, "by_kind": by_kind, "missed": missed,
            "false_positives": fps, "findings": len(r.findings), "score": r.overall_score, "verdict": r.verdict,
            "mode": r.mode, "seconds": r.total_duration_s}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--min-recall", type=float, default=0.0)
    ap.add_argument("--max-fp", type=int, default=10 ** 6)
    ap.add_argument("--out", default=None, help="записать markdown-таблицу результатов")
    a = ap.parse_args()
    cfg = LLMConfig.from_env()
    files = sorted((ROOT / "eval").glob("*.expected.json"))
    results = [evaluate(f, a.llm and cfg.enabled, cfg) for f in files]

    rows = ["| ТЗ | Режим | Найдено | Полнота | Ложные | Находок | Балл |", "|---|---|---|---|---|---|---|"]
    for res in results:
        rows.append(f"| {Path(res['sample']).name} | {res['mode']} | {res['detected']}/{res['total']} | "
                    f"{res['recall']:.0%} | {len(res['false_positives'])} | {res['findings']} | {res['score']} |")
    tot_d = sum(r["detected"] for r in results)
    tot_t = sum(r["total"] for r in results)
    tot_fp = sum(len(r["false_positives"]) for r in results)
    recall = tot_d / tot_t if tot_t else 1.0
    rows.append(f"| **Итого** | | **{tot_d}/{tot_t}** | **{recall:.0%}** | **{tot_fp}** | | |")
    kinds: dict[str, list[int]] = {}
    for res in results:
        for k, (d, t) in res["by_kind"].items():
            kinds.setdefault(k, [0, 0])
            kinds[k][0] += d
            kinds[k][1] += t
    rows += ["", "| Тип дефекта | Найдено | Полнота |", "|---|---|---|"]
    rows += [f"| {k} | {d}/{t} | {d / t:.0%} |" for k, (d, t) in sorted(kinds.items())]
    print("\n".join(rows))
    for res in results:
        if res["missed"] or res["false_positives"]:
            print(f"\n{res['sample']}:")
            for m in res["missed"]:
                print("  пропущено:", m)
            for fp in res["false_positives"]:
                print("  ложное:", fp)
    if a.out:
        Path(a.out).write_text("\n".join(rows) + "\n", encoding="utf-8")
    if recall < a.min_recall or tot_fp > a.max_fp:
        print(f"\nFAIL: recall {recall:.0%} < {a.min_recall:.0%} или ложных {tot_fp} > {a.max_fp}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
