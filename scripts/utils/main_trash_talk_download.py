#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import requests

# ВСТАВЬТЕ СЮДА СВОЙ API-КЛЮЧ С FREESOUND
FREESOUND_API_TOKEN = "VQMTDfKUao7ZWfVWrWKsZkLyn7agPAAadzuWvrhX"


BASE_URL = "https://freesound.org/apiv2"


def get_random_sound(query: str, filter_str: str | None = None) -> dict:
    """
    Возвращает один случайный звук по текстовому запросу.
    Можно передать filter_str для доп. фильтрации.
    """
    params = {
        "query": query,
        "sort": "random",
        "page_size": 1,
        "fields": "id,name,previews,license,username",
        "token": FREESOUND_API_TOKEN,
    }
    if filter_str:
        params["filter"] = filter_str

    resp = requests.get(f"{BASE_URL}/search/text/", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("count", 0) == 0:
        raise RuntimeError(
            f"По запросу '{query}' (filter={filter_str!r}) ничего не найдено"
        )

    return data["results"][0]


def download_file(url: str, out_path: Path):
    """
    Скачивает файл по URL.
    """
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def sanitize_filename(name: str) -> str:
    """
    Удаляет из имени файла «опасные» символы.
    """
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")
    return name.strip()


def download_random_sounds(kind: str, n: int, out_dir: Path):
    """
    kind: 'speech', 'noise' или 'both'
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- РЕЧЬ: ИЩЕМ АНГЛ. ФРАЗЫ ПРО РУССКУЮ РЕЧЬ --------
    if kind in ("speech", "both"):
        # только query, без filter tag:"russian"
        speech_queries = [
            "russian speech",
            "russian voice",
            "russian talking",
            "russian conversation",
            "russian dialog",
        ]
    else:
        speech_queries = []

    # -------- ШУМ: как раньше --------
    if kind in ("noise", "both"):
        noise_queries = [
            ("noise", None),
            ("ambience", None),
            ("background noise", None),
            ("room tone", None),
        ]
    else:
        noise_queries = []

    # приводим всё к формату (query, filter)
    query_pairs: list[tuple[str, str | None]] = []
    for q in speech_queries:
        query_pairs.append((q, None))
    query_pairs.extend(noise_queries)

    if not query_pairs:
        raise ValueError("Нужно указать тип: speech, noise или both")

    downloaded = 0
    attempts = 0
    max_attempts = n * 10  # запас попыток, чтобы пробовать разные запросы/рандом

    while downloaded < n and attempts < max_attempts:
        attempts += 1
        query, flt = query_pairs[attempts % len(query_pairs)]

        print(f"[{downloaded+1}/{n}] Попытка {attempts}: query={query!r}, filter={flt!r}...")

        try:
            sound = get_random_sound(query, filter_str=flt)
        except RuntimeError as e:
            print("  ", e)
            continue

        sound_id = sound["id"]
        name = sanitize_filename(sound["name"])
        user = sound["username"]
        license_ = sound["license"]

        # mp3-превью
        preview_url = sound["previews"].get("preview-hq-mp3") \
                      or sound["previews"].get("preview-lq-mp3")
        if not preview_url:
            print("  Нет mp3-превью, пропускаю.\n")
            continue

        # добавляем индекс, чтобы файлы не перезаписывались
        filename = f"{downloaded+1}_{sound_id}_{name}.mp3"
        out_path = out_dir / filename

        print(f"  Скачиваю: {preview_url}")
        print(f"  В файл:   {out_path}")
        print(f"  Автор:    {user}, лицензия: {license_}")
        try:
            download_file(preview_url, out_path)
        except Exception as e:
            print(f"  Ошибка при скачивании: {e}\n")
            continue

        downloaded += 1
        time.sleep(2)  # чтобы не долбить API слишком часто
        print("  Готово.\n")

    if downloaded < n:
        print(f"Скачано только {downloaded} файлов из {n} — больше подходящих "
              f"звуков по запросам не нашлось (или много ошибок).")


def main():
    parser = argparse.ArgumentParser(
        description="Скачивание случайных звуков речи/шума с Freesound.org"
    )
    parser.add_argument(
        "--type",
        choices=["speech", "noise", "both"],
        default="both",
        help="тип звуков: речь (speech), шум (noise) или оба варианта (both)",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=5,
        help="сколько звуков скачать",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="downloads",
        help="папка для сохранения звуков",
    )

    args = parser.parse_args()

    if FREESOUND_API_TOKEN == "YOUR_FREESOUND_API_KEY":
        raise SystemExit(
            "Сначала получите API-ключ на https://freesound.org/apiv2/apply/ "
            "и вставьте его в переменную FREESOUND_API_TOKEN."
        )

    out_dir = Path(args.out)
    download_random_sounds(args.type, args.num, out_dir)


if __name__ == "__main__":
    main()