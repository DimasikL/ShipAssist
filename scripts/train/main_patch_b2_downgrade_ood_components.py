"""
patch_b2_downgrade_ood_components.py — Приоритет B2.

Что делает:
  • §2.5: EnsembleOutlierGate и WADA-SNR переводятся из основного
    описания системы в «направления дальнейших исследований».
  • §3.5: блок Hybrid C+ / HybridAudioEngine заменяется нейтральной
    фразой о расширяемости конвейера.
  • §2.5 [210]: добавляет ссылку на таблицу выбора порога (§4.4, табл. 4.4).

Скрипт идемпотентен: если правки уже применены — сообщает об этом
и не трогает файл.

Запуск (из корня ShipAssistant/):
    python scripts/train/patch_b2_downgrade_ood_components.py

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
# Каждая пара — точная строка из документа.
# Если old не найден — значит правка уже применена, пропускаем.
# ─────────────────────────────────────────────────────────────────────────────

PATCHES = [

    # §2.5 [210] — добавить ссылку на таблицу 4.4
    (
        'что подтверждено экспериментально (§4.4).',
        'что подтверждено экспериментально (§4.4, таблица 4.4).',
        '§2.5 [210]: добавлена ссылка на таблицу 4.4'
    ),

    # §2.5 [211] — EnsembleOutlierGate → упоминание как направления развития
    (
        'В расширенной реализации (Сценарий 1) базовый одиночный детектор заменён '
        'ансамблевым шлюзом EnsembleOutlierGate, объединяющим три метрики: расстояние '
        'Махаланобиса (вес 2), косинусное расстояние (вес 1) и евклидово L2 (вес 1). '
        'Каждая метрика нормализуется робастным z-преобразованием (медиана + IQR-масштаб) '
        'и суммируется: s = 0,5·z_mahal + 0,25·z_cos + 0,25·z_L2. Отклонение при s > τ_c, '
        'где τ_c откалиброван для каждого класса на 95-м перцентиле внутриклассовых баллов — '
        'компактные кластеры получают жёсткий порог, разреженные — мягкий, без дополнительного '
        'обучения. Реализация: core/hybrid/outlier_gate.py, класс EnsembleOutlierGate.',
        'Реализованный детектор является базовым одиночным детектором Махаланобиса. '
        'В качестве направления дальнейших исследований (§5.5) рассматривается ансамблевый '
        'шлюз EnsembleOutlierGate (core/hybrid/outlier_gate.py), объединяющий расстояние '
        'Махаланобиса с косинусным расстоянием и L2-нормой — для повышения устойчивости '
        'при расширении словаря команд.',
        '§2.5 [211]: EnsembleOutlierGate понижен до направления развития'
    ),

    # §2.5 [212] — WADA-SNR → прототип
    (
        'Дополнительно применяется адаптация порога к уровню шума по приближению '
        'WADA-SNR (Hirsch & Pearce, 2000): ОСШ оценивается из спектрограммы мощности '
        'скользящим минимумом/максимумом в окне 400 мс. Порог корректируется по формуле: '
        'τ_адап = τ₀ + β · max(0, ОСШ_реф − ОСШ_online), где β = 0,15; ОСШ_реф = 12 дБ. '
        'В чистых условиях (ОСШ ≥ 12 дБ) порог не меняется; при зашумлённом сигнале — '
        'возрастает, снижая риск ложной активации. Реализация: функции estimate_snr_db() '
        'и snr_adaptive_threshold() в core/hybrid/outlier_gate.py.',
        'Адаптация порога к текущему уровню шума по методу WADA-SNR (Hirsch & Pearce, 2000) '
        'реализована прототипно в core/hybrid/outlier_gate.py и также отнесена к '
        'направлениям дальнейших исследований (§5.5).',
        '§2.5 [212]: WADA-SNR понижен до прототипа'
    ),

    # §3.5 [244] — Hybrid C+ / HybridAudioEngine → нейтральная фраза
    (
        'В режиме Hybrid C+ (HybridAudioEngine) конвейер расширяется четырёхэтапным '
        'маршрутом для каждого аудиокадра. Этап 1 — извлечение эмбеддинга: '
        'переиспользуется тензор outputs[1] ONNX-сессии (нулевые накладные расходы) '
        'или запускается автономный WTVEmbedder. Этап 2 — шлюз OOD (EnsembleOutlierGate): '
        'при отклонении возвращается словарь с outlier_rejected=True, обработка прекращается. '
        'Этап 3 — косинусный поиск центроида (CentroidSearch): классификация по ближайшему '
        'центроиду в нормированном пространстве эмбеддингов. Этап 4 — заполнение слота: '
        'при совпадении с одним из slot_intents запускается числовой регрессор MLP '
        '(Вариант А: NumberRegressor) или CTC-декодер (Вариант Б: CTCDigitDecoder); '
        'метод фиксируется в поле slot_method для A/B-телеметрии.',
        'Реализованный конвейер поддерживает расширение через подключаемые модули: '
        'прототип расширенного маршрута с ансамблевым OOD-шлюзом и числовым регрессором '
        'описан в §5.5 как направление дальнейших исследований.',
        '§3.5 [244]: убран блок Hybrid C+ / HybridAudioEngine'
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────────────────────

def _replace(para, old: str, new: str) -> bool:
    """Заменить old→new в параграфе. Возвращает True если замена произошла."""
    full = ''.join(r.text for r in para.runs)
    if old not in full:
        return False
    new_full = full.replace(old, new)
    # Попытка заменить внутри одного рана (сохраняет форматирование)
    for run in para.runs:
        if old in run.text:
            run.text = run.text.replace(old, new)
            return True
    # Текст разбит по ранам — сворачиваем в первый
    para.runs[0].text = new_full
    for run in para.runs[1:]:
        run.text = ''
    return True


def apply_patches(doc, dry_run: bool = False) -> list[str]:
    """Применить все замены. Вернуть список выполненных меток."""
    done, skipped = [], []
    for old, new, label in PATCHES:
        applied = False
        for para in doc.paragraphs:
            if old in ''.join(r.text for r in para.runs):
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

    p = argparse.ArgumentParser(description='B2: понизить статус TTA/EnsembleOutlierGate')
    p.add_argument('--docx',    default=str(_ROOT / 'VKR_Lucher_v5.docx'),
                   help='Входной .docx (по умолчанию VKR_Lucher_v5.docx)')
    p.add_argument('--out',     default=None,
                   help='Выходной .docx (по умолчанию — перезаписать входной)')
    p.add_argument('--dry-run', action='store_true',
                   help='Показать что изменится, не записывать файл')
    args = p.parse_args()

    docx_path = Path(args.docx)
    out_path  = Path(args.out) if args.out else docx_path

    if not docx_path.exists():
        print(f'Ошибка: файл не найден: {docx_path}', file=sys.stderr)
        sys.exit(1)

    doc = Document(str(docx_path))
    done, skipped = apply_patches(doc, dry_run=args.dry_run)

    print(f'{"[DRY-RUN] " if args.dry_run else ""}Применено ({len(done)}/{len(PATCHES)}):')
    for label in done:
        print(f'  ✅ {label}')
    if skipped:
        print(f'Уже применено / не найдено ({len(skipped)}):')
        for label in skipped:
            print(f'  — {label}')

    if not args.dry_run:
        doc.save(str(out_path))
        print(f'\nСохранено: {out_path}')


if __name__ == '__main__':
    main()
