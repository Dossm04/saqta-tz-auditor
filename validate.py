"""Агент проверки требований (ISO/IEC/IEEE 29148, IEEE 830) — запуск одной командой.

    python validate.py <файл требований: .pdf | .docx | .md | .txt> <файл отчёта: .md>

Примеры:
    python validate.py docs/requirements.pdf reports/report.md
    python validate.py docs/requirements.md reports/report.md --offline     # без LLM (только правила)
    python validate.py docs/requirements.pdf reports/report.md --json reports/report.json --model gemini-3-flash-preview

Ключ LLM берётся из .env (см. .env.example): GEMINI_API_KEY или LLM_API_KEY/LLM_BASE_URL/LLM_MODEL.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agents.iso_review import review_document  # noqa: E402
from config import LLMConfig  # noqa: E402
from parsing import load_path  # noqa: E402
from report_iso import to_markdown  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Статическая проверка документа требований по ISO/IEC/IEEE 29148 и IEEE 830")
    p.add_argument("input", help="документ требований (.pdf, .docx, .md, .txt)")
    p.add_argument("output", help="куда записать отчёт (.md)")
    p.add_argument("--offline", action="store_true", help="без LLM: только детерминированные правила")
    p.add_argument("--model", help="переопределить модель LLM")
    p.add_argument("--json", help="дополнительно сохранить отчёт в JSON")
    a = p.parse_args()

    src = Path(a.input)
    if not src.exists():
        print(f"Файл не найден: {src}", file=sys.stderr)
        return 2
    cfg = LLMConfig.from_env().with_overrides(model=a.model)
    use_llm = not a.offline
    if use_llm and not cfg.enabled:
        print("Ключ LLM не найден (.env: GEMINI_API_KEY или LLM_API_KEY) — работаю на правилах. Для полной проверки добавьте ключ.")
    doc = load_path(src)
    print(f"Документ: {doc.name}, строк: {doc.non_empty_count}; модель: {cfg.model if use_llm and cfg.enabled else 'офлайн-правила'}")

    def on_event(agent: str, phase: str, note: str = "") -> None:
        if phase == "done":
            print(f"  ✓ {agent:<10} {note}")

    report = review_document(doc, cfg, use_llm=use_llm, on_event=on_event)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(report), encoding="utf-8")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    s = report.stats
    print(f"\nТребований: {s['requirements']}; нарушений: {s['issues_total']} "
          f"(критичных {s['critical']}, важных {s['major']}, мелких {s['minor']}); режим: {report.mode}")
    print(f"Итог: {report.verdict}")
    print(f"Отчёт: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
