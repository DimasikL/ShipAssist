"""
verify_ring_buffer.py — Верификация параметров кольцевого буфера (§2.3 ВКР).

Назначение: воспроизводимое подтверждение расчётных значений C_min и кратности
запаса буфера, используемых в §2.3, §2.8 и Заключении ВКР.

Usage:
    python scripts/vkr/verify_ring_buffer.py

Expected output:
    C_min = 7 952 отсч., запас ≈ 20.1×
"""

from __future__ import annotations


def verify_ring_buffer(
    s_in: float = 16_000.0,
    t_inf: float = 0.247,
    t_jitter: float = 0.05,
    t_margin: float = 0.20,
    capacity: int = 160_000,
) -> dict[str, float]:
    """Рассчитать минимальную ёмкость кольцевого буфера и запас.

    Args:
        s_in: Частота дискретизации входного аудиопотока, отсч./с.
        t_inf: Время инференса ONNX-модели, с (247 мс → 0.247).
        t_jitter: Оценка джиттера планировщика ОС, с.
        t_margin: Технологический запас безопасности, с.
        capacity: Принятая ёмкость буфера в реализации, отсч.

    Returns:
        Словарь с промежуточными и итоговыми значениями.
    """
    t_total: float = t_inf + t_jitter + t_margin
    c_min: float = s_in * t_total
    safety_margin: float = capacity / c_min

    return {
        "S_in (отсч./с)": s_in,
        "T_inf (с)": t_inf,
        "T_jitter (с)": t_jitter,
        "T_margin (с)": t_margin,
        "T_total = T_inf + T_jitter + T_margin (с)": t_total,
        "C_min = S_in × T_total (отсч.)": c_min,
        "C_принятое (отсч.)": float(capacity),
        "Запас = C / C_min (×)": safety_margin,
    }


def main() -> None:
    """Точка входа: вывод расчёта на стандартный поток."""
    results = verify_ring_buffer()

    print("=" * 60)
    print("Верификация параметров кольцевого буфера (§2.3 ВКР)")
    print("=" * 60)
    for name, value in results.items():
        if "отсч." in name:
            print(f"  {name:<45} = {value:>10,.0f}")
        elif "×" in name:
            print(f"  {name:<45} = {value:>10.2f}")
        else:
            print(f"  {name:<45} = {value:>10.4f}")
    print("=" * 60)

    c_min = results["C_min = S_in × T_total (отсч.)"]
    margin = results["Запас = C / C_min (×)"]

    # Assertions — «автотест» для CI/pre-defence check
    assert abs(c_min - 7_952.0) < 1.0, (
        f"ОШИБКА: C_min = {c_min:.1f}, ожидалось 7 952"
    )
    assert 20.0 <= margin <= 21.0, (
        f"ОШИБКА: запас = {margin:.2f}×, ожидалось ≈ 20×"
    )

    print(f"\n✓ C_min = {c_min:,.0f} отсч.  |  Запас = {margin:.1f}× (≈ 20×)")
    print("  Все значения согласованы с §2.3, §2.8 и Заключением (задача 6).")


if __name__ == "__main__":
    main()
