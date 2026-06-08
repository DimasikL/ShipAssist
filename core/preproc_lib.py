from scipy import signal

def equalizer(audio, sr):
    nyquist = sr / 2.0
    lowcut = 100.0
    highcut = 3000.0
    order = 6
    b, a = signal.butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    filtered_audio = signal.filtfilt(b, a, audio)
    return filtered_audio
