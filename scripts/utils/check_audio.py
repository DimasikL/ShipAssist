"""Quick audio sanity check for misclassified files."""
import librosa
import numpy as np

files = [
    r"clf_dset/test/group=train_user_2/samples/приготовить_машину_x5/untitled.wav",
    r"clf_dset/test/group=train_user_2/samples/приготовить_машину_x5/untitled-2.wav",
    r"clf_dset/test/group=train_user_2/samples/приготовить_машину_x5/untitled-3.wav",
    r"clf_dset/test/group=train_user_2/samples/приготовить_машину_x5/untitled-4.wav",
    r"clf_dset/test/group=train_user_2/samples/приготовить_машину_x5/untitled-5.wav",
    r"clf_dset/test/group=train_user_2/samples/самый_малый_вперед_x5/untitled-2.wav",
    r"clf_dset/test/group=train_user_2/samples/самый_малый_вперед_x5/untitled-3.wav",
    r"clf_dset/test/group=train_user_2/samples/самый_малый_вперед_x5/untitled-4.wav",
    r"clf_dset/test/group=train_user_2/samples/самый_малый_вперед_x5/untitled-5.wav",
]

print(f"{'file':25s}  {'duration':>10s}  {'rms':>10s}  {'max_amp':>10s}")
print("-" * 65)
for f in files:
    fname = f.split("/")[-1]
    try:
        y, sr = librosa.load(f, sr=16000)
        rms = float(np.sqrt(np.mean(y ** 2)))
        max_amp = float(np.abs(y).max())
        duration = len(y) / sr
        flag = "  ⚠ SILENT" if rms < 0.001 else ("  ⚠ SHORT" if duration < 0.5 else "")
        print(f"{fname:25s}  {duration:>10.2f}s  {rms:>10.5f}  {max_amp:>10.5f}{flag}")
    except Exception as e:
        print(f"{fname:25s}  ERROR: {e}")
