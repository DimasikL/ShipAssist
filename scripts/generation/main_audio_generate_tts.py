import os
from pathlib import Path

import requests
import torch
import torchaudio
from gtts import gTTS
from joblib import Parallel, delayed
from tqdm import tqdm
from TTS.api import TTS

SILERO_MODEL_FILE = 'model_v4.pt'
DEVICE = torch.device('cpu')
torch.set_num_threads(8)
MODEL_SAMPLE_RATE = 24000
TARGET_SAMPLE_RATE = 16000
SILERO_SPEAKERS = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene']


# def generate_tts(text: str, output_path: str):
#    model_name = "tts_models/en/ljspeech/vits"
#    # model_name = "tts_models/ru/v3_1_ru/model"
#    tts = TTS(model_name)
#    tts.tts_to_file(text=text, file_path=output_path)

def generate_silero(text: str, output_path: str, speaker='baya', model=None, resampler=None):
    audio = model.apply_tts(text=text, speaker=speaker, sample_rate=MODEL_SAMPLE_RATE)
    if isinstance(audio, list):
        audio = audio[0]
    audio_16k = resampler(audio)
    torchaudio.save(output_path, audio_16k.unsqueeze(0), TARGET_SAMPLE_RATE)


def generate_gtts(text: str, output_path: str):
    tts = gTTS(text=text, lang='ru')
    tts.save(output_path)


def generate_ytts(text: str, output_path: str, api_key: str, voice='alyss'):
    url = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"

    headers = {"Authorization": f"Api-Key {api_key}"}

    data = {
        "text": text,
        "lang": "ru-RU",
        "voice": voice,
        "format": "mp3"
    }

    response = requests.post(url, headers=headers, data=data, stream=True)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Аудиофайл сохранен в {output_path}")
    else:
        print(f"Ошибка: {response.status_code} - {response.text}")


def generate_vits(text: str, output_path: str, model_name: str = "tts_models/ru/v3_1_ru/vits"):
    tts = TTS(model_name)
    tts.tts_to_file(text=text, file_path=output_path)


def main():
    # from TTS.utils.manage import ModelManager
    # manager = ModelManager()
    # models = manager.list_models()
    # print("Доступные модели:")
    # for model in models:
    #     if '/ru/' in model:
    #         print(model)

    commands = [
        'машина', 'приготовить машину', 'самый малый вперед',
        #     *[f"поворот {direction} на {angle}" for direction in ['влево', 'вправо'] for angle in range(5, 31)]
    ]
    # postf = ''
    # commands = [
    #    "лес", "яркое солнце", "глубокая тишина", "синий свет", "ночной город",
    #    "тихий океан", "старый замок", "мягкий плед", "горячий чай", "прохладный ветер"]  # ,
    #     "спелое яблоко", "дождливая осень", "веселый смех", "ледяной дождь", "утренний кофе",
    #     "сладкий сон", "темный лес", "зеленый мох", "плотная бумага", "острый нож",
    #     "игра теней", "золотой песок", "большой дом", "старинная карта", "хрустальный звон",
    #     "детская радость", "снежная вершина", "ночное небо", "белая рубашка", "грубая ткань",
    #     "камень", "шелест листвы", "крепкий чай", "красное вино", "вечерняя прогулка",
    #     "песня ветра", "пыльная дорога", "желтый свет", "стеклянная ваза", "мягкий снег",
    #     "скорый поезд", "вечный огонь", "осенний лес", "чистая вода", "тонкий лёд",
    #     "зелёный луг", "старый рюкзак", "коричневый мишка", "глухая тайга", "новая звезда",
    #     "тихая гавань", "громкий звук", "простая мысль", "нежный взгляд", "вкусный пирог",
    #     "снежный день", "жаркий полдень", "тёмное пиво", "лунный свет", "радужный мост",
    #     "звонкий голос", "тёплый вечер", "яркий фонарь", "освежающий бриз", "чёрный кофе",
    #     "рыжая кошка", "ледяная скала", "шум прибоя", "широкая улыбка", "короткий путь",
    #     "новая жизнь", "длинная дорога", "неоновый свет", "влажный песок", "пряный аромат",
    #     "зимний лес", "утренний туман", "каменный мост", "жёлтые листья", "полная луна",
    #     "лёгкий ветер", "мелкий дождь", "спокойное море", "быстрый бег", "глубокий сон",
    #     "лёгкая грусть", "морская пена", "вечерняя заря", "прохладная роса", "детский смех",
    #     "гладкий камень", "новая книга", "старая дверь", "зелёное яблоко", "нежный шёпот",
    #     "тихая песня", "забытый сон", "лунная дорожка", "безмолвный лес", "радостный день",
    #     "вечный путь", "прозрачная вода", "ночная тишина", "теплый плед", "морозный воздух"
    # ]
    # postf = 'другие слова'
    # commands = [str(n) for n in range(5, 31)]
    postf = ''

    model_name = 'gtts'
    # res_dir = Path('mfcc_db_numbers/tts_5_30')
    res_dir = Path(f'clf_dset/train_val/group={model_name}/samples')
    res_dir.mkdir(exist_ok=True, parents=True)

    if model_name == 'silero':
        if not os.path.isfile(SILERO_MODEL_FILE):
            torch.hub.download_url_to_file(
                'https://models.silero.ai/models/tts/ru/v4_ru.pt',
                SILERO_MODEL_FILE
            )
        model = torch.package.PackageImporter(SILERO_MODEL_FILE).load_pickle("tts_models", "model")
        model.to(DEVICE)
        resampler = torchaudio.transforms.Resample(orig_freq=MODEL_SAMPLE_RATE, new_freq=TARGET_SAMPLE_RATE)

        for speaker in SILERO_SPEAKERS:
            out_dir = res_dir / f'{model_name}_{speaker}'
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, cmd in enumerate(commands):
                output_path = str(out_dir / f'{model_name} {cmd}_{speaker}.wav')
                generate_silero(
                    text=cmd,
                    output_path=output_path,
                    speaker=speaker,
                    model=model,
                    resampler=resampler
                )




    elif model_name == 'gtts':
        out_dir = res_dir / f'{model_name}'
        out_dir.mkdir(parents=True, exist_ok=True)
        Parallel(n_jobs=4)(
            delayed(generate_gtts)(
                text=cmd,
                output_path=str(out_dir / f'{model_name} {cmd}.wav')
            )
            for i, cmd in enumerate(tqdm(commands))
        )

    elif model_name == 'ytts':
        out_dir = res_dir / f'{model_name}'
        out_dir.mkdir(parents=True, exist_ok=True)
        yandex_api_key = os.getenv('YANDEX_API_KEY')  # Установи переменную окружения или впиши вручную
        Parallel(n_jobs=4)(
            delayed(generate_ytts)(
                text=cmd,
                output_path=str(out_dir / f'{model_name}_{cmd} {postf}.mp3'),
                api_key=yandex_api_key
            )
            for cmd in tqdm(commands)
        )



if __name__ == '__main__':
    main()
