import os
import csv
import re
import tarfile
import requests
from io import BytesIO
from tqdm import tqdm

API_KEY = "fa041b65746549dd7978a337bdc83bab43799816f8c6c85f1bde24a54c4ae516"
DATASET_ID = "cmflnuzw70j7lpa4qgjzcbuyu"   # ID русского датасета
SAVE_DIR = "data_mozilla"
TSV_NAME = "train.tsv"  # можно validated.tsv

TARGET_PHRASES = [
    "машина", "приготовить", "машину", "вперед", "самый",
    "малый", "лес", "солнце", "тишина", "свет",
    "город", "океан", "замок", "плед", "дождь", "кофе",
]

MAX_PER_PHRASE = 50

os.makedirs(SAVE_DIR, exist_ok=True)

def normalize(s: str) -> str:
    return re.sub(r"[^\wа-яё]", "", s.lower()).strip()

targets_norm = [normalize(t) for t in TARGET_PHRASES]

# 1) создаём download-session
resp = requests.post(
    f"https://datacollective.mozillafoundation.org/api/datasets/{DATASET_ID}/download",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
)
if resp.status_code != 200:
    print("Ошибка создания загрузочной сессии:", resp.status_code, resp.text)
    exit()

session = resp.json()
download_url = session["downloadUrl"]
total_size = session.get("sizeBytes", None)
print("Скачиваем архив:", session["filename"])

# 2) скачиваем архив потоково в память
buffer = BytesIO()
with requests.get(download_url, headers={"Authorization": f"Bearer {API_KEY}"}, stream=True) as r:
    if r.status_code != 200:
        print("Ошибка скачивания:", r.status_code, r.text)
        exit()

    total = int(r.headers.get("Content-Length", 0))
    chunk_count = total // 8192 if total else None

    for chunk in tqdm(r.iter_content(chunk_size=8192), total=chunk_count):
        buffer.write(chunk)

buffer.seek(0)

# 3) извлекаем только нужный TSV в память и собираем matching-файлы
path_to_phrases = {}          # относительный путь → какие фразы подходят
collected = {p: 0 for p in TARGET_PHRASES}

with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
    members = tar.getmembers()

    # ищем наш train.tsv внутри архива
    tsv_member = None
    for m in members:
        if m.name.endswith(TSV_NAME):
            tsv_member = m
            break

    if not tsv_member:
        print("TSV файл не найден:", TSV_NAME)
        exit()

    print("Читаем TSV:", tsv_member.name)
    tsv_f = tar.extractfile(tsv_member)
    reader = csv.DictReader((line.decode("utf-8") for line in tsv_f), delimiter="\t")

    for row in reader:
        sentence = row.get("sentence") or row.get("text") or row.get("transcript") or ""
        s_norm = normalize(sentence)
        for idx, t_norm in enumerate(targets_norm):
            if t_norm in s_norm:
                relative = row.get("path") or ""
                if relative:
                    path_to_phrases.setdefault(relative, set()).add(TARGET_PHRASES[idx])

    if not path_to_phrases:
        print("В TSV нет совпадений для TARGET_PHRASES.")
        exit()

    # создаём папки
    for ph in TARGET_PHRASES:
        os.makedirs(os.path.join(SAVE_DIR, ph.replace(" ", "_")), exist_ok=True)

    print(f"Найдено {len(path_to_phrases)} нужных файлов. Извлекаем только их...")

    # теперь извлечём только соответствующие аудио
    for member in tqdm(members, desc="Аудиофайлы"):
        if not member.isreg():
            continue
        if not member.name.endswith(".mp3"):
            continue

        # member.name типа: clips/ru/common_voice_ru_123456.mp3
        filename = os.path.basename(member.name)

        # по ключу TSV путь обычно без 'clips/'
        for relative, phrases in path_to_phrases.items():
            if filename == os.path.basename(relative):
                # сохраняем этот файл для всех совпавших фраз (как в твоем коде)
                mf = tar.extractfile(member)
                data = mf.read()

                for ph in phrases:
                    if collected[ph] >= MAX_PER_PHRASE:
                        continue

                    out_dir = os.path.join(SAVE_DIR, ph.replace(" ", "_"))
                    out_name = f"{os.path.splitext(filename)[0]}_{collected[ph]:04d}.mp3"
                    out_path = os.path.join(out_dir, out_name)

                    with open(out_path, "wb") as f:
                        f.write(data)

                    collected[ph] += 1

    print("Готово!")
    for ph, cnt in collected.items():
        print(f" {ph}: {cnt}")
