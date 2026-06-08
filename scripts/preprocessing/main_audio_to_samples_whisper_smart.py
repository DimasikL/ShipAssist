import os
import re
from pathlib import Path

import whisper
from pydub import AudioSegment
from tqdm import tqdm

from core.embedders import CommonEmbedder


def get_audio_by_time(audio, start_time: float, end_time: float) -> AudioSegment:
    start_ms = int(start_time * 1000)
    end_ms = int(end_time * 1000)

    return audio[start_ms:end_ms]


def split_audio_into_whisper_batches(model, audio_path, audio_dir, pref, prompt):
    audio_dir.mkdir(exist_ok=True)

    audio = AudioSegment.from_file(audio_path)
    result = model.transcribe(
        str(audio_path),
        language='ru',
        task="transcribe",
        condition_on_previous_text=False,
        prompt=prompt
    )

    segments = result['segments']

    for i, segment in tqdm(enumerate(segments)):
        text = segment['text'].lower()
        text = re.sub(r'[^\w\s]', '', text)
        # print(f"[{segment['start']:.2f} - {segment['end']:.2f}] {text}")
        duration = segment['end'] - segment['start']

        audio_batch = get_audio_by_time(audio=audio, start_time=segment['start'], end_time=segment['end'])

        audio_batch_path = audio_dir.joinpath(f"{i} {pref} duration={duration:.2f},{text}.wav")

        audio_batch.set_channels(1).set_frame_rate(16000).set_sample_width(2)

        audio_batch.export(audio_batch_path, format='wav')


def split_and_validate(model, base_path, prompt):
    pref = base_path.stem

    res_dir = base_path.parent.joinpath(base_path.stem)

    split_audio_into_whisper_batches(
        model=model,
        audio_path=base_path,
        audio_dir=res_dir,
        pref=pref,
        prompt=prompt
    )

    embedder = CommonEmbedder(sr=16000, my_prep=True, norm_duration=True)

    # TODO split by timestamps
    # first_split_files = [Path(f"{dir}/{file}") for dir, _, files in os.walk(f'audio_to_cut/{pref}') for file in files]

    # for file in tqdm(first_split_files):
    #     assert file.exists(), file
    #     duration = re.search(r'duration=([\d.]+)', file.name).group(1)
    #     duration = float(duration)
    #
    #     if duration > 6:
    #         split_audio_into_whisper_batches(model=model, audio_path=file, audio_dir=res_dir, pref=pref)
    #         os.remove(file)

    final_split_files = [Path(f"{dir}/{file}") for dir, _, files in os.walk(f'audio_to_cut/{pref}') for file in
                         files]

    for file in tqdm(final_split_files):
        try:
            embedder.get_prep_waveform(file)
        except Exception as e:
            print('removed because of preproc exception')
            os.remove(file)


def main():
    model = whisper.load_model("large", device='cpu')

    base_paths = [
        # Path("audio_to_cut/povorot vlevo 5-30 1.wav"),
        # Path("audio_to_cut/machina i td.m4a"),
        # Path("audio_to_cut/machina i td s problemami.m4a"),
        # Path('audio_to_cut/danila-whisper-2/machina i td.wav')
        # Path('audio_to_cut/danila-whisper-2/povorot vlevo 5-12.wav'),
        # Path('audio_to_cut/danila-whisper-2/povorot vpravo 5-12.wav'),
        # Path('audio_to_cut/danila-whisper-2/machina i td.wav'),
        Path('audio_to_cut/danila-whisper-2/povorot vlevo 5-13.wav'),
        Path('audio_to_cut/danila-whisper-2/povorot vpravo 5-13.wav')

    ]
    for path in base_paths:
        assert isinstance(path, Path)
    for base_path in base_paths:
        split_and_validate(
            model=model,
            base_path=base_path,
            prompt="""
                Ожидается команда вида "поворот влево на УГОЛ" либо "поворот вправо на УГОЛ"
                Угол принимает значения 5-30
                Команды не подходящие под шаблон стоит игнорировать
            """
            # prompt="""
            #     Ожидается команда вида "машина" либо "приготовить машину" либо "самый малый вперед"
            #     Фрагменты аудио не подходящие под шаблон стоит помечать как "другие слова"
            #     Распознанные фразы должны в точности подходить под указанные шаблоны, нельзя использовать ё вместо е
            # """
        )


if __name__ == '__main__':
    main()
