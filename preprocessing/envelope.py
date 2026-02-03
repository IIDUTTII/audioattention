import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly

def extract_envelope(audio, fs, target_fs=64, n_bands=28, verbose=False):
    if fs != 8000: audio = resample_poly(audio, 8000, fs); fs = 8000
    cf_min, cf_max = 150, 4000
    erb_low = 21.4*np.log10(4.37*cf_min/1000+1); erb_high = 21.4*np.log10(4.37*cf_max/1000+1)
    cfs = (10**(np.linspace(erb_low, erb_high, n_bands)/21.4)-1)/4.37*1000
    envs = []; total = len(cfs)
    for i, cf in enumerate(cfs, 1):
        bw = 1.5*24.7*(4.37*cf/1000+1)
        low = max(cf-bw/2, 20)
        high = min(cf+bw/2, fs/2-1)
        if high <= low: continue
        band = sosfiltfilt(butter(4,[low,high],btype="band",fs=fs,output="sos"), audio)
        envs.append(np.abs(band))
        if verbose: print(f"\rBands processed: {i}/{total}", end="")
    if verbose: print()
    if not envs: raise RuntimeError("No valid bands")
    env = np.mean(envs,axis=0)**0.6
    return resample_poly(env, target_fs, fs) if fs != target_fs else env