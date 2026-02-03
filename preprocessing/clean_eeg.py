import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly

def preprocess_eeg(eeg, fs, target_fs=64, bandpass=(0.5, 32), rereference=True, verbose=False):
    sos = butter(4, bandpass, btype="band", fs=fs, output="sos")
    eeg = sosfiltfilt(sos, eeg, axis=1)
    if verbose: print(f"Filtered {bandpass[0]}–{bandpass[1]} Hz")

    if fs != target_fs:
        eeg = resample_poly(eeg, target_fs, fs, axis=1)
        if verbose: print(f"Downsampled {fs} → {target_fs} Hz")

    if rereference:
        eeg = eeg - eeg.mean(axis=0, keepdims=True)
        if verbose: print("Average re-referenced")

    eeg = eeg - eeg.mean(axis=1, keepdims=True)
    if verbose: print("Zero-centered")

    return eeg
