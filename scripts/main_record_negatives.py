"""
scripts/record_negatives.py — запись негативных примеров для класса "другие слова".

Сохраняет клипы как {слово}_{NNN:03d}.wav.
При повторном запуске — продолжает нумерацию, не перезаписывает.

Использование:
    python scripts\\record_negatives.py
    python scripts\\record_negatives.py --duration 2.0 --neg_dir clf_dset\\negatives
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path


# ── Список слов ────────────────────────────────────────────────────────────────
#
# Цель: покрыть краевые случаи, при которых модель ошибочно детектирует команду.
# Команды: "машина" | "приготовить машину" | "самый малый вперёд"
#
# Формат: (текст, длительность_сек)
# Длинные предложения — 3.5с, фразы — 2.5с, одиночные слова — 1.5с
WORDS: list[tuple[str, float]] = [
    # ── 1. Предложения С командными словами внутри ────────────────────────────────
    # Главный краевой случай: слово команды в бытовом контексте.
    # Говори НОРМАЛЬНО, как в разговоре — не как команду.
    ("моя машина сломалась",          3.5),
    ("поставь машину в гараж",        3.5),
    ("купи новую машину",             3.0),
    ("это твоя машина",               2.5),
    ("вызови машину скорой",          3.5),
    ("надо приготовить ужин",         3.5),
    ("я уже приготовил",              2.5),
    ("приготовить завтрак не успею",  4.0),
    ("иди вперёд по улице",           3.0),
    ("смотри вперёд",                 2.0),
    ("самый маленький из всех",       3.5),
    ("это самый лучший вариант",      3.5),
    ("на малой скорости",             3.0),

    # ── 2. Фонетически близкие к "машина" ────────────────────────────────────────
    ("малина",     1.5),
    ("Марина",     1.5),
    ("резина",     1.5),
    ("корзина",    1.5),
    ("вершина",    1.5),
    ("пружина",    1.5),
    ("машинка",    1.5),

    # ── 3. Фонетически близкие к "приготовить" ───────────────────────────────────
    ("подготовить",      2.0),
    ("приготовиться",    2.0),
    ("готовить",         1.5),
    ("готово",           1.5),
    ("готовься",         1.5),
    ("приготовить ужин", 2.5),
    ("приготовить кашу", 2.5),

    # ── 4. Фонетически близкие к "самый малый вперёд" ────────────────────────────
    ("самый маленький",  2.5),
    ("самый большой",    2.5),
    ("самый быстрый",    2.5),
    ("малый вперёд",     2.0),
    ("самый малый",      2.0),
    ("полный вперёд",    2.0),
    ("малый назад",      2.0),
    ("средний вперёд",   2.0),

    # ── 5. Фрагменты команд (частичное произнесение) ─────────────────────────────
    ("машину",       1.5),
    ("приготовить",  1.5),
    ("самый",        1.5),
    ("малый",        1.5),
    ("вперёд",       1.5),

    # ── 6. Другие морские команды (не в 4 классах) ────────────────────────────────
    ("стоп машина",  2.0),
    ("полный назад", 2.0),
    ("стоп",         1.5),
    ("товсь",        1.5),
    ("левый борт",   2.0),
    ("правый борт",  2.0),

    # ── 7. Числа (одной фразой — экономит время) ─────────────────────────────────
    ("раз два три",              2.0),
    ("четыре пять шесть",        2.0),
    ("семь восемь девять десять", 2.5),
    ("ноль один два",            2.0),

    # ── 9. Поток речи — паузы, отмены, разговорный темп ──────────────────────────
    # Говори быстро и естественно, как прерывая себя.
    ("э-э-э подожди",   2.5),
    ("нет не то",        2.0),
    ("ладно давай",      2.0),
    ("подожди минуту",   2.5),
    ("ещё раз",          1.5),

    # ── 10. Эмоциональная/быстрая речь ───────────────────────────────────────────
    # Говори раздражённо или быстро — как в стрессовой ситуации.
    ("нет стоп подожди",      2.5),
    ("да да всё понял",       2.5),
    ("не сейчас потом",       2.5),
    ("всё отставить",         2.0),

    # ── 11. Тихая речь (имитация фона) ───────────────────────────────────────────
    # Говори намеренно ТИХО — вполголоса, как будто не к системе.
    # Это покрывает краевые случаи с низкой энергией (~1e-4).
    ("тихо приготовить",       2.0),   # произноси тихо: "приготовить"
    ("тихо вперёд малый",      2.5),   # произноси тихо: "вперёд малый"
    ("тихо самый малый",       2.5),   # произноси тихо: "самый малый"

    # ── 12. Короткая бытовая речь ─────────────────────────────────────────────────
    ("да",           1.5),
    ("нет",          1.5),
    ("хорошо",       1.5),
    ("понятно",      1.5),
    ("спасибо",      1.5),
    ("пожалуйста",   1.5),

    ("открой дверь", 2.5),
    ("который час",  2.5),

    ("тест проверка", 2.0),
]

# ── Команды для дозаписи (позитивные примеры) ─────────────────────────────────
# Записываются в отдельные папки по классу: clf_dset/commands/{класс}/
# Цель: покрыть вариации — тихо, быстро, с расстояния, с паузами.
# Папки: машина / приготовить_машину / самый_малый_вперёд
#
# Инструкции по стилю указаны в скобках — произноси соответственно.
COMMANDS: list[tuple[str, str, float]] = [
    # (класс, подсказка для записи, длительность)

    # ── машина ───────────────────────────────────────────────────────────────────
    ("машина", "машина  [нормально]",         2.0),
    ("машина", "машина  [громко, чётко]",     2.0),
    ("машина", "машина  [тихо, вполголоса]",  2.0),
    ("машина", "машина  [быстро]",            1.5),
    ("машина", "машина  [медленно]",          2.5),
    ("машина", "машина  [отдалённо от mic]",  2.0),

    # ── приготовить машину ────────────────────────────────────────────────────────
    ("приготовить_машину", "приготовить машину  [нормально]",        3.0),
    ("приготовить_машину", "приготовить машину  [громко, чётко]",    3.0),
    ("приготовить_машину", "приготовить машину  [тихо, вполголоса]", 3.0),
    ("приготовить_машину", "приготовить машину  [быстро]",           2.5),
    ("приготовить_машину", "приготовить машину  [медленно]",         3.5),
    ("приготовить_машину", "приготовить машину  [отдалённо от mic]", 3.0),

    # ── самый малый вперёд ────────────────────────────────────────────────────────
    ("самый_малый_вперёд", "самый малый вперёд  [нормально]",        3.0),
    ("самый_малый_вперёд", "самый малый вперёд  [громко, чётко]",    3.0),
    ("самый_малый_вперёд", "самый малый вперёд  [тихо, вполголоса]", 3.0),
    ("самый_малый_вперёд", "самый малый вперёд  [быстро]",           2.5),
    ("самый_малый_вперёд", "самый малый вперёд  [медленно]",         3.5),
    ("самый_малый_вперёд", "самый малый вперёд  [отдалённо от mic]", 3.0),
]


# ── audio ──────────────────────────────────────────────────────────────────────

def _write_wav(path: str, raw_int16: bytes, sr: int = 16000) -> None:
    n = len(raw_int16)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", n))
        f.write(raw_int16)


def _warmup_stream(sr: int = 16000) -> None:
    """Open and immediately close a dummy stream to prime the audio driver.

    sounddevice takes ~0.3–1 s to initialise the device on the first call to
    sd.rec().  Calling this once before the recording loop eliminates the
    silent gap at the start of every subsequent recording.
    """
    import sounddevice as sd

    # Record a tiny throwaway buffer — just enough to force driver init.
    sd.rec(int(0.05 * sr), samplerate=sr, channels=1, dtype="int16", blocking=True)


def _record(duration: float, sr: int = 16000) -> bytes:
    """Record *duration* seconds of 16-bit mono audio at *sr* Hz.

    Assumes the audio driver has already been warmed up via :func:`_warmup_stream`.
    """
    import sounddevice as sd

    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1,
                   dtype="int16", blocking=True)
    return audio.tobytes()


# ── helpers ────────────────────────────────────────────────────────────────────

def _slug(word: str) -> str:
    """Turn word into a safe filename stem (keep Cyrillic, replace spaces)."""
    s = word.strip().replace(" ", "_").replace("-", "_")
    # remove chars that are problematic on Windows
    s = re.sub(r'[\\/:*?"<>|]', "", s)
    return s


def _next_idx(out_dir: Path, slug: str) -> int:
    """Return next free index for this slug (continues existing files)."""
    existing = list(out_dir.glob(f"{slug}_*.wav"))
    if not existing:
        return 1
    indices = []
    for p in existing:
        m = re.search(r"_(\d+)\.wav$", p.name)
        if m:
            indices.append(int(m.group(1)))
    return max(indices) + 1 if indices else 1


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["negatives", "commands", "all"],
                   default="all",
                   help="negatives = другие слова | commands = команды | all = оба (default)")
    p.add_argument("--duration", type=float, default=2.0,
                   help="Fallback duration if not specified per item (default 2.0)")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--neg_dir", type=Path, default=Path("clf_dset/negatives"),
                   help="Output dir for negatives (default: clf_dset/negatives)")
    p.add_argument("--cmd_dir", type=Path, default=Path("clf_dset/commands"),
                   help="Output dir for commands (default: clf_dset/commands)")
    p.add_argument("--reps", type=int, default=2,
                   help="Repetitions per item (default 2)")
    return p.parse_args()


def _record_loop(
    items: list,
    out_dir: Path,
    reps: int,
    sr: int,
    default_dur: float,
    section_label: str,
    item_fmt,  # callable(item, default_dur) -> (display_hint, slug, duration)
) -> int:
    """Generic recording loop. Returns number of clips saved."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total_saved = 0

    print(f"\n{'=' * 58}")
    print(f"  {section_label}  ({len(items)} items × {reps} reps)")
    print(f"  Output: {out_dir.resolve()}")
    print(f"  Enter = record  |  s = skip item  |  q = quit")
    print(f"{'=' * 58}")

    for item in items:
        hint, slug, dur = item_fmt(item, default_dur)

        for rep in range(1, reps + 1):
            idx = _next_idx(out_dir, slug)
            prompt = f"  [{hint}  rep {rep}/{reps}  {dur}s] > "
            try:
                key = input(prompt).strip().lower()
            except (KeyboardInterrupt, EOFError):
                print(f"\nStopped. Saved {total_saved} clips.")
                return total_saved

            if key == "q":
                print(f"Quit. Saved {total_saved} clips.")
                sys.exit(0)
            if key == "s":
                print(f"    Skipped '{hint}'")
                break

            fname = f"{slug}_{idx:03d}.wav"
            fpath = out_dir / fname
            print(f"    Recording {dur}s...", end="", flush=True)
            try:
                raw = _record(dur, sr)
            except Exception as e:
                print(f" ERROR: {e}")
                continue
            _write_wav(str(fpath), raw, sr)
            print(f" -> {fname}")
            total_saved += 1

    return total_saved


def main() -> None:
    args = parse_args()

    # Prime the audio driver once so subsequent sd.rec() calls start instantly.
    print("Warming up audio device...", end=" ", flush=True)
    _warmup_stream(args.sr)
    print("ready.")

    def neg_fmt(item, default_dur):
        if isinstance(item, tuple):
            word, dur = item
        else:
            word, dur = item, default_dur
        return word, _slug(word), dur

    total_saved = 0

    if args.mode in ("negatives", "all"):
        saved = _record_loop(
            items=WORDS,
            out_dir=args.neg_dir,
            reps=args.reps,
            sr=args.sr,
            default_dur=args.duration,
            section_label="НЕГАТИВЫ (другие слова)",
            item_fmt=neg_fmt,
        )
        total_saved += saved

    if args.mode in ("commands", "all"):
        # Group COMMANDS by class slug so we record one class at a time.
        classes: dict[str, list] = {}
        for entry in COMMANDS:
            cls_slug = entry[0]
            classes.setdefault(cls_slug, []).append(entry)

        for cls_slug, entries in classes.items():
            cls_dir = args.cmd_dir / cls_slug

            def cmd_fmt_cls(item, _default_dur):
                _, hint, dur = item
                return hint, _slug(hint.split("[")[0].strip()), dur

            saved = _record_loop(
                items=entries,
                out_dir=cls_dir,
                reps=args.reps,
                sr=args.sr,
                default_dur=args.duration,
                section_label=f"КОМАНДЫ / {cls_slug}",
                item_fmt=cmd_fmt_cls,
            )
            total_saved += saved

    print(f"\nВсего сохранено: {total_saved} клипов.")


if __name__ == "__main__":
    main()
