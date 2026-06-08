from pydub import AudioSegment


def convert_m4a_to_wav(m4a_path, wav_path, sr):
    audio = AudioSegment.from_file(m4a_path)
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)  # 2 bytes = 16-bit
    audio.export(wav_path, format="wav")
