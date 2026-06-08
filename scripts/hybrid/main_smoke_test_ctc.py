"""
scripts/hybrid/smoke_test_ctc.py — Проверка что CTCDigitDecoder работает без артефактов.

Тест синтетический: используем внутренние функции декодера,
убеждаемся что decode() возвращает число в разумном диапазоне.

Запуск:
    python scripts/hybrid/smoke_test_ctc.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np

from core.hybrid.ctc_digit_decoder import (
    CTCDigitDecoder,
    VOCAB_SIZE,
    BLANK_IDX,
    _VOCAB,
    _TOKEN_TO_VALUE,
    WORD_TO_TOKEN,
    _greedy_ctc_decode,
    _tokens_to_int,
)


def test_vocab() -> None:
    """Проверить что словарь содержит ожидаемые токены."""
    # VOCAB_SIZE = 34 (33 слова + 1 blank)
    assert VOCAB_SIZE == 34, f"Expected 34 tokens (33 words + blank), got {VOCAB_SIZE}"

    # blank занимает индекс 0
    assert BLANK_IDX == 0, f"BLANK_IDX must be 0, got {BLANK_IDX}"

    # spot-checks: слова присутствуют в словаре
    words_in_vocab = {word for _, (word, _) in _VOCAB.items()}
    for expected in ("ноль", "двести", "триста", "тридцать", "восемьдесят"):
        assert expected in words_in_vocab, f"'{expected}' missing from vocab"

    # additive values
    assert _TOKEN_TO_VALUE[WORD_TO_TOKEN["сто"]]         == 100
    assert _TOKEN_TO_VALUE[WORD_TO_TOKEN["двести"]]      == 200
    assert _TOKEN_TO_VALUE[WORD_TO_TOKEN["двадцать"]]    ==  20
    assert _TOKEN_TO_VALUE[WORD_TO_TOKEN["восемьдесят"]] ==  80
    assert _TOKEN_TO_VALUE[WORD_TO_TOKEN["пять"]]        ==   5

    print("✓ Vocabulary test passed  "
          f"(VOCAB_SIZE={VOCAB_SIZE}, BLANK_IDX={BLANK_IDX})")


def test_tokens_to_value() -> None:
    """Проверить что конкретные последовательности токенов дают правильные числа."""
    # "двести восемьдесят пять" = 200 + 80 + 5 = 285
    tokens_285 = [
        WORD_TO_TOKEN["двести"],
        WORD_TO_TOKEN["восемьдесят"],
        WORD_TO_TOKEN["пять"],
    ]
    assert _tokens_to_int(tokens_285) == 285, (
        f"Expected 285, got {_tokens_to_int(tokens_285)}"
    )

    # "сто" = 100
    assert _tokens_to_int([WORD_TO_TOKEN["сто"]]) == 100

    # "тридцать" = 30
    assert _tokens_to_int([WORD_TO_TOKEN["тридцать"]]) == 30

    # "двадцать один" = 20 + 1 = 21
    tokens_21 = [WORD_TO_TOKEN["двадцать"], WORD_TO_TOKEN["один"]]
    assert _tokens_to_int(tokens_21) == 21, (
        f"Expected 21, got {_tokens_to_int(tokens_21)}"
    )

    # пустой список → None
    assert _tokens_to_int([]) is None, "Empty token list must return None"

    print("✓ Token-to-value test passed  (285, 100, 30, 21, None)")


def test_greedy_ctc_decode_synthetic() -> None:
    """Проверить greedy CTC decode на синтетических log-probs.

    Создаём log_probs так, чтобы для нескольких фреймов argmax указывал
    на конкретные токены, и проверяем collapse-логику.
    """
    T = 10
    # Все фреймы — blank (0), кроме двух окон: idx 2-3 → "сто" (31), idx 6-7 → "два" (4)
    log_probs = np.full((T, VOCAB_SIZE), -10.0, dtype=np.float32)
    log_probs[:, BLANK_IDX] = 0.0           # по умолчанию blank везде
    log_probs[2, WORD_TO_TOKEN["сто"]] = 1.0
    log_probs[3, WORD_TO_TOKEN["сто"]] = 1.0  # повтор — должен collapse
    log_probs[6, WORD_TO_TOKEN["два"]] = 1.0
    log_probs[7, WORD_TO_TOKEN["два"]] = 1.0  # повтор — должен collapse

    decoded = _greedy_ctc_decode(log_probs)
    assert decoded == [WORD_TO_TOKEN["сто"], WORD_TO_TOKEN["два"]], (
        f"Expected [сто, два], got {[_VOCAB.get(t, ('?',))[0] for t in decoded]}"
    )

    value = _tokens_to_int(decoded)
    assert value == 102, f"Expected 100+2=102, got {value}"

    print(f"✓ Greedy CTC decode test passed  "
          f"(decoded=[сто, два] → {value})")


def test_ctc_decoder_not_loaded() -> None:
    """Проверить что predict() безопасно возвращает (None, 0.0) если head не загружен."""
    decoder = CTCDigitDecoder(frame_dim=64, min_val=0, max_val=360)
    assert not decoder._is_loaded, "Freshly constructed decoder must have _is_loaded=False"

    frames = np.random.randn(20, 64).astype(np.float32)
    val, conf = decoder.predict(frames)

    assert val is None, f"Expected None when not loaded, got {val}"
    assert conf == 0.0, f"Expected 0.0 confidence when not loaded, got {conf}"

    print("✓ Not-loaded guard test passed  (val=None, conf=0.0)")


def test_ctc_decoder_with_synthetic_head() -> None:
    """Полный end-to-end: создать head вручную, загрузить в decoder и вызвать predict().

    Имитирует CTCDigitDecoder.load() без файловой системы — напрямую подключаем head.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    frame_dim = 64
    T = 20

    # Создать LinearHead и заменить _head._net
    decoder = CTCDigitDecoder(frame_dim=frame_dim, min_val=0, max_val=360)
    linear = nn.Linear(frame_dim, VOCAB_SIZE)
    torch.nn.init.normal_(linear.weight, std=0.01)
    decoder._head._net = linear
    decoder._is_loaded = True   # имитируем успешную загрузку

    frames = np.random.randn(T, frame_dim).astype(np.float32)
    val, conf = decoder.predict(frames)

    # val может быть None если greedy decode вернул пустую последовательность —
    # это нормально при случайных весах (blank может доминировать).
    # Главное: нет исключений и типы корректные.
    if val is not None:
        assert isinstance(val, float), f"val must be float, got {type(val)}"
        assert 0.0 <= val <= 360.0, f"val={val} out of [0, 360]"
        assert 0.0 <= conf <= 1.0,  f"conf={conf} out of [0, 1]"
        print(f"✓ Synthetic head test passed  (val={val:.1f}, conf={conf:.3f})")
    else:
        # blank доминирует — корректное поведение при случайных весах
        assert conf == 0.0, "conf must be 0.0 when val is None"
        print(f"✓ Synthetic head test passed  (val=None — blank dominant, conf={conf:.3f})")


if __name__ == "__main__":
    print("=== CTCDigitDecoder Smoke Test ===\n")

    test_vocab()
    test_tokens_to_value()
    test_greedy_ctc_decode_synthetic()
    test_ctc_decoder_not_loaded()
    test_ctc_decoder_with_synthetic_head()

    print("\n✓ All smoke tests passed")
