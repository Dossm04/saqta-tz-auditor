"""Полнота проверки по ISO 29148 на документе кейса относительно экспертной разметки.

    python scripts/eval_iso.py                 # офлайн-правила
    python scripts/eval_iso.py --llm           # LLM + правила (нужен ключ)
    python scripts/eval_iso.py --report r.json # оценить уже сформированный JSON-отчёт (validate.py --json)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.iso_review import IsoReport, review_document  # noqa: E402
from config import LLMConfig  # noqa: E402
from parsing import load_path  # noqa: E402


# какие характеристики засчитываются для типа дефекта
KIND_CHARS = {
    "contradiction": {"set_consistent", "correct"},
    "incomplete": {"complete", "set_complete", "conforming", "missing"},
    "missing": {"set_complete", "complete", "missing"},
    "singular": {"singular"},
    "not_atomic": {"singular"},
    "unverifiable": {"verifiable", "set_verifiable", "unambiguous"},
    "acceptance": {"set_verifiable", "set_complete", "set_consistent", "verifiable", "complete", "missing"},
    "ambiguity": {"unambiguous", "complete", "verifiable"},
    "verifiable": {"verifiable", "set_verifiable", "set_validatable", "unambiguous", "correct", "set_consistent"},
    "set": {"set_complete", "set_modifiable", "set_traceable", "set_verifiable", "set_ranked", "set_comprehensible",
            "set_validatable", "conforming"},
    "security": {"security", "set_consistent", "correct", "appropriate"},
}


def _mentions(report: IsoReport) -> list[tuple[set[int], str, str]]:
    """(строки, текст, характеристика) для каждого нарушения/пропуска отчёта."""
    lines_of = {u.rid: set(u.lines) for u in report.units}
    out = []
    for u in report.units:
        for i in u.issues:
            ls = set(i.lines) | set(u.lines)
            for r in i.related:
                ls |= lines_of.get(r, set())
            out.append((ls, " ".join([i.problem, i.fix, *i.evidence]).lower(), i.characteristic))
    sr = report.set_review
    for i in sr.issues:
        ls = set(i.lines)
        for r in i.related:
            ls |= lines_of.get(r, set())
        out.append((ls, " ".join([i.problem, i.fix, *i.evidence]).lower(), i.characteristic))
    for m in sr.missing:
        ls = set()
        for r in m.get("related") or []:
            ls |= lines_of.get(r, set())
        out.append((ls, " ".join(str(m.get(k, "")) for k in ("topic", "why", "suggested_requirement")).lower(), "missing"))
    return out


def evaluate(report: IsoReport, expected: dict) -> tuple[int, int, list[str]]:
    mentions = _mentions(report)
    found, missed = 0, []
    for d in expected["defects"]:
        exp_lines = set(d["lines"])
        kws = [k.lower() for k in d.get("keywords", [])]
        chars = KIND_CHARS.get(d["kind"], set())
        ok = any(ls & exp_lines and (not chars or ch in chars) and (not kws or any(k in txt for k in kws))
                 for ls, txt, ch in mentions)
        if ok:
            found += 1
        else:
            missed.append(f"{d['id']} [{d['kind']}] {d['description']}")
    return found, len(expected["defects"]), missed


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--report", help="готовый JSON-отчёт validate.py --json (по .md документа кейса)")
    ap.add_argument("--min-recall", type=float, default=0.0)
    ap.add_argument("--blind", action="store_true", help="слепые документы eval/iso/blind/*.json вместо документа кейса")
    ap.add_argument("--split", choices=["dev", "holdout", "all"], default="all", help="какие слепые документы брать")
    a = ap.parse_args()
    if a.blind:
        cfg = LLMConfig.from_env()
        files = sorted((ROOT / "eval" / "iso" / "blind").glob("*.json"))
        if a.split != "all":
            files = [f for f in files if f.stem.split("_")[0].rstrip("0123456789") == a.split]
        tf = tt = 0
        for f in files:
            exp = json.loads(f.read_text(encoding="utf-8"))
            rep = review_document(load_path(ROOT / exp["sample"]), cfg, use_llm=a.llm and cfg.enabled)
            found, total, missed = evaluate(rep, exp)
            tf, tt = tf + found, tt + total
            print(f"{f.stem}: {found}/{total} ({found / total:.0%}); режим: {rep.mode}; нарушений: {rep.stats['issues_total']}")
            for m in missed:
                print("    пропущено:", m)
        print(f"ИТОГО слепые документы ({a.split}): {tf}/{tt} ({tf / max(tt, 1):.0%})")
        return 0 if tt and tf / tt >= a.min_recall else (0 if not tt else 1)
    expected = json.loads((ROOT / "eval" / "iso" / "case_requirements.json").read_text(encoding="utf-8"))
    if a.report:
        report = IsoReport.model_validate_json(Path(a.report).read_text(encoding="utf-8"))
    else:
        cfg = LLMConfig.from_env()
        report = review_document(load_path(ROOT / expected["sample"]), cfg, use_llm=a.llm and cfg.enabled)
    found, total, missed = evaluate(report, expected)
    print(f"Документ кейса: найдено {found}/{total} экспертных дефектов ({found / total:.0%}); режим: {report.mode}; "
          f"нарушений в отчёте: {report.stats['issues_total']}")
    for m in missed:
        print("  пропущено:", m)
    return 0 if found / total >= a.min_recall else 1


if __name__ == "__main__":
    raise SystemExit(main())
