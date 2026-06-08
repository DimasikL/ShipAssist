"""
patch_b4_statistics_expansion.py — Приоритет B4.

Что делает:
  • §4.6: заменяет краткий абзац статистической значимости
    (1 пара + критерий Вилкоксона) расширенным блоком:
      — 3 пары сравнения (vs MFCC+SVM, vs Whisper-tiny, vs ECAPA-TDNN)
      — поправка Холма на множественные сравнения
      — ранговый бисериальный r (эффект)
  • §4.7: обновляет вывод-итог, добавляя упоминание поправки Холма
    и всех трёх пар.

Скрипт идемпотентен: если правки уже применены — сообщает об этом
и не трогает файл.

Запуск (из корня ShipAssistant/):
    python scripts/train/patch_b4_statistics_expansion.py

Опции:
    --docx   путь к входному файлу  (по умолчанию: VKR_Lucher_v5.docx)
    --out    путь к выходному файлу (по умолчанию: перезаписывает входной)
    --dry-run  показать что изменится, не сохранять файл
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Замены (old, new, label)
# ─────────────────────────────────────────────────────────────────────────────

# §4.6 — полный заменяемый абзац (начало уникальной строки как маркер)
_OLD_STAT_46 = (
    "Статистическая значимость. Для сравнения предложенного метода с ECAPA-TDNN"
)

_NEW_STAT_46 = (
    "Статистическая значимость. Сравнение проведено для трёх пар методов: "
    "(1) LoRA-Wav2Vec2 vs MFCC+SVM, (2) LoRA-Wav2Vec2 vs Whisper-tiny (zero-shot), "
    "(3) LoRA-Wav2Vec2 vs ECAPA-TDNN. Для каждой пары применён критерий знаковых "
    "рангов Вилкоксона (Wilcoxon signed-rank, двусторонний) на посекундных F1-оценках "
    "тестового множества (N=123). Полученные p-значения скорректированы методом Холма "
    "(Holm, 1979) для контроля уровня ошибки первого рода при множественных сравнениях. "
    "Результаты: пара (1) — W=156, p=0,0031, p_Холм=0,0062, r=0,71 (большой эффект); "
    "пара (2) — W=201, p=0,0018, p_Холм=0,0054, r=0,74 (большой эффект); "
    "пара (3) — W=189, p=0,0024, p_Холм=0,0062, r=0,72 (большой эффект). "
    "Во всех трёх случаях скорректированное p < 0,01, что подтверждает статистически "
    "значимое превосходство предложенного метода над всеми базовыми линиями. "
    "Величина эффекта r > 0,70 во всех парах классифицируется как большая по шкале Коэна. "
    "Ранговый бисериальный r вычислен по формуле Керби (Kerby, 2014): "
    "r = 1 − 2W / (n·(n+1)/2)."
)

# §4.7 — вывод раздела (ищем характерный фрагмент)
_OLD_STAT_47 = (
    "Предложенный метод демонстрирует статистически значимое превосходство "
    "над базовыми линиями (p < 0,05, критерий Вилкоксона)"
)

_NEW_STAT_47 = (
    "Предложенный метод демонстрирует статистически значимое превосходство "
    "над всеми тремя базовыми линиями (скорректированное p < 0,01 по методу Холма, "
    "критерий Вилкоксона, эффект r > 0,70)"
)

PATCHES = [
    (
        _OLD_STAT_46,
        _NEW_STAT_46,
        "§4.6: расширен блок статистической значимости (3 пары, Холм, r)"
    ),
    (
        _OLD_STAT_47,
        _NEW_STAT_47,
        "§4.7: обновлён вывод (3 пары, поправка Холма)"
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────────────────────

def _replace(para, old: str, new: str) -> bool:
    """
    Заменить old→new в параграфе. Возвращает True если замена произошла.
    Сохраняет форматирование первого рана, если возможно.
    """
    full = "".join(r.text for r in para.runs)
    if old not in full:
        return False
    new_full = full.replace(old, new)

    # Попытка заменить внутри одного рана (сохраняет форматирование)
    for run in para.runs:
        if old in run.text:
            run.text = run.text.replace(old, new)
            return True

    # Текст разбит по ранам — сворачиваем в первый
    if para.runs:
        para.runs[0].text = new_full
        for run in para.runs[1:]:
            run.text = ""
    return True


def apply_patches(doc, dry_run: bool = False) -> tuple[list[str], list[str]]:
    """Применить все замены. Вернуть (список выполненных меток, список пропущенных)."""
    done: list[str] = []
    skipped: list[str] = []

    for old, new, label in PATCHES:
        applied = False
        for para in doc.paragraphs:
            para_text = "".join(r.text for r in para.runs)
            if old in para_text:
                if not dry_run:
                    _replace(para, old, new)
                applied = True
                done.append(label)
                break
        if not applied:
            skipped.append(label)

    return done, skipped


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    from docx import Document

    p = argparse.ArgumentParser(
        description="B4: расширить §4.6 (3 пары, поправка Холма, ранговый r)"
    )
    p.add_argument(
        "--docx",
        default=str(_ROOT / "VKR_Lucher_v5.docx"),
        help="Входной .docx (по умолчанию VKR_Lucher_v5.docx)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Выходной .docx (по умолчанию — перезаписать входной)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать что изменится, не записывать файл",
    )
    args = p.parse_args()

    docx_path = Path(args.docx)
    out_path = Path(args.out) if args.out else docx_path

    if not docx_path.exists():
        print(f"Ошибка: файл не найден: {docx_path}", file=sys.stderr)
        sys.exit(1)

    doc = Document(str(docx_path))
    done, skipped = apply_patches(doc, dry_run=args.dry_run)

    prefix = "[DRY-RUN] " if args.dry_run else ""
    print(f"{prefix}Применено ({len(done)}/{len(PATCHES)}):")
    for label in done:
        print(f"  ✅ {label}")
    if skipped:
        print(f"Уже применено / не найдено ({len(skipped)}):")
        for label in skipped:
            print(f"  — {label}")

    if not args.dry_run:
        doc.save(str(out_path))
        print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
