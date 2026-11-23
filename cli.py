"""CLI: python cli.py samples/sample_tz.md [--offline] [--out report.md] [--json report.json]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agents.orchestrator import analyze
from config import LLMConfig
from parsing import load_path
from report import to_json, to_markdown


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Мульти-агентная проверка качества финтех-ТЗ")
    p.add_argument("file", help="ТЗ: .md / .txt / .docx / .pdf")
    p.add_argument("--offline", action="store_true", help="только правила, без LLM")
    p.add_argument("--model", help="переопределить LLM_MODEL")
    p.add_argument("--out", help="сохранить Markdown-отчёт")
    p.add_argument("--json", help="сохранить JSON-отчёт")
    a = p.parse_args()

    cfg = LLMConfig.from_env().with_overrides(model=a.model)
    doc = load_path(a.file)

    def on_event(kind, agent, run):
        if kind == "done":
            print(f"  [{run.stage}] {agent.title:<20} {run.mode:<8} {run.duration_s:>6.2f}s  → {run.produced:>3}  {run.note}")

    use_llm = not a.offline
    if use_llm and not cfg.enabled:
        print("LLM_API_KEY не задан — работаю в офлайн-режиме (правила).")
    print(f"Анализ «{doc.name}» ({len(doc.lines)} строк)…")
    r = analyze(doc, cfg, use_llm=use_llm, on_event=on_event)
    sc = r.severity_counts()
    cov, part, total = r.control_coverage()
    print(f"\nИТОГ: {r.overall_score}/100 — {r.verdict}")
    print(f"Находки: 🔴 {sc['critical']}  🟠 {sc['major']}  🟡 {sc['minor']} | вопросов: {len(r.questions)} | "
          f"контроли: {cov}+{part}½/{total}")
    for d in r.dimensions:
        print(f"  {d.name:<26} {d.score:>3}")
    md = to_markdown(r)
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
        print(f"Markdown: {a.out}")
    if a.json:
        Path(a.json).write_text(to_json(r), encoding="utf-8")
        print(f"JSON: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
