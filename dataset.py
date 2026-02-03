# dataset.py - COMPLETE VERSION
import os, sys, pickle, hashlib, numpy as np, torch, scipy.io
from torch.utils.data import Dataset
from scipy.io import wavfile
from scipy.signal import resample_poly
from joblib import Parallel, delayed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.clean_eeg import preprocess_eeg
from preprocessing.envelope import extract_envelope


def window_signal(x, win):
    n = x.shape[-1] // win
    return x[..., :n * win].reshape(*x.shape[:-1], n, win)


class KULDataset(Dataset):
    def __init__(self, subject_file, stimuli_dir, window_length=10.0, fs_target=64, 
                 use_cache=True, cache_dir="cache", verbose=True, n_jobs=8, 
                 training=True):
        self.subject_file = subject_file
        self.stimuli_dir = stimuli_dir
        self.window_length = window_length
        self.fs_target = fs_target
        self.window_samples = int(window_length * fs_target)
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.training = training

        if verbose:
            print(f"\nLoading subject: {os.path.basename(subject_file)}")
            print(f"Window: {window_length}s @ {fs_target} Hz")

        cache = self._cache_path(cache_dir)

        if use_cache and os.path.exists(cache):
            if verbose: print("Loading from cache")
            self._load_cache(cache)
        else:
            if verbose: print("Processing from scratch")
            self._process_all_trials()
            if use_cache: self._save_cache(cache)

        if verbose: print(f"Dataset ready: {len(self.labels)} windows")

    def _cache_path(self, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        key = f"{self.subject_file}_{self.window_length}_{self.fs_target}"
        return os.path.join(cache_dir, os.path.basename(self.subject_file).replace(".mat", "") + "_" + hashlib.md5(key.encode()).hexdigest()[:8] + ".pkl")

    def _process_all_trials(self):
        trials = scipy.io.loadmat(self.subject_file)["trials"][0]
        out = Parallel(n_jobs=self.n_jobs)(delayed(self._process_trial)(t) for t in trials)
        
        # Handle different trial lengths
        eeg_list = []
        audio_list = []
        label_list = []
        
        for eeg, aud, lab in out:
            if eeg.shape[0] > 0:
                eeg_list.append(eeg)
                audio_list.append(aud)
                label_list.append(lab)
        
        self.eeg_windows = np.concatenate(eeg_list, axis=0)
        self.audio_windows = np.concatenate(audio_list, axis=0)
        self.labels = np.concatenate(label_list, axis=0).astype(np.int64)

    def _process_trial(self, trial):
        try:
            fs_eeg = int(trial["FileHeader"][0][0]["SampleRate"][0][0][0][0])
            label = 0 if str(trial["attended_ear"][0][0][0]) == "L" else 1

            eeg = preprocess_eeg(trial["RawData"][0][0]["EegData"][0][0].T, fs_eeg, self.fs_target)

            stim = trial["stimuli"][0][0]
            fs_a, al = wavfile.read(os.path.join(self.stimuli_dir, str(stim[0][0][0])))
            _, ar = wavfile.read(os.path.join(self.stimuli_dir, str(stim[1][0][0])))

            al, ar = al.astype(np.float32)/32768.0, ar.astype(np.float32)/32768.0
            el, er = extract_envelope(al, fs_a, fs_eeg), extract_envelope(ar, fs_a, fs_eeg)

            if fs_eeg != self.fs_target:
                el, er = resample_poly(el, self.fs_target, fs_eeg), resample_poly(er, self.fs_target, fs_eeg)

            T = min(eeg.shape[1], len(el), len(er))
            eeg, aud = eeg[:, :T], np.stack([el[:T], er[:T]])

            # ========== FIX: Window then transpose ==========
            # eeg shape: (64, T)
            # aud shape: (2, T)
            
            # Calculate number of complete windows
            n_windows = T // self.window_samples
            
            if n_windows == 0:  # Skip trials too short
                return np.empty((0, 64, self.window_samples), dtype=np.float32), \
                    np.empty((0, 2, self.window_samples), dtype=np.float32), \
                    np.empty(0, dtype=np.int64)
            
            # Trim to exact window length
            T_trim = n_windows * self.window_samples
            eeg = eeg[:, :T_trim]
            aud = aud[:, :T_trim]
            
            # Reshape: (channels, n_windows, window_samples)
            eeg_w = eeg.reshape(64, n_windows, self.window_samples)
            aud_w = aud.reshape(2, n_windows, self.window_samples)
            
            # Transpose to: (n_windows, channels, window_samples)
            eeg_w = eeg_w.transpose(1, 0, 2)
            aud_w = aud_w.transpose(1, 0, 2)
            # ===============================================

            # Standardize each windowhttps://scontent.cdninstagram.com/v/t51.71878-15/503157340_1366025924613877_6484422668238776315_n.jpg?stp=dst-jpg_e15_tt6&_nc_cat=101&ig_cache_key=MzM2MDkxMTE3NjEwMjU4ODAzNw==.3-ccb7-5&ccb=7-5&_nc_sid=58cdad&efg=eyJ2ZW5jb2RlX3RhZyI6InhwaWRzLjEwODB4MTkyMC5zZHIuQzMifQ==&_nc_ohc=pkaphgx6qpUQ7kNvwF_Zp3-&_nc_oc=AdkDarPkumBYc3Q5EUWQh_cm_eR2IaC1YGs67hUggjlDY0kUvkN17LdlPRpfyWJk249K6NyEnyQQ2pfFaZP6POnt&_nc_zt=23&_nc_ht=scontent.cdninstagram.com&_nc_gid=irptr2aTF-LwihIJ919EXQ&oh=00_Aftf-jBjdt37qHBfNeNF1VAgWJghHVOrhQ32zp0X4xoL4g&oe=69858565

            eeg_w = (eeg_w - eeg_w.mean((1,2), keepdims=True)) / (eeg_w.std((1,2), keepdims=True) + 1e-8)
            aud_w = (aud_w - aud_w.mean((1,2), keepdims=True)) / (aud_w.std((1,2), keepdims=True) + 1e-8)

            return eeg_w.astype(np.float32), aud_w.astype(np.float32), np.full(n_windows, label)
        
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️ Trial failed: {e}")
            return np.empty((0, 64, self.window_samples), dtype=np.float32), \
                np.empty((0, 2, self.window_samples), dtype=np.float32), \
                np.empty(0, dtype=np.int64)


    def _save_cache(self, p):
        """Save processed data to cache"""
        with open(p, "wb") as f: 
            pickle.dump({
                "eeg": self.eeg_windows, 
                "audio": self.audio_windows, 
                "labels": self.labels
            }, f)

    def _load_cache(self, p):
        """Load from cache"""
        with open(p, "rb") as f: 
            d = pickle.load(f)
        self.eeg_windows = d["eeg"]
        self.audio_windows = d["audio"]
        self.labels = d["labels"]

    def __len__(self): 
        return len(self.labels)

    def __getitem__(self, i):
        eeg = torch.from_numpy(self.eeg_windows[i])
        audio = torch.from_numpy(self.audio_windows[i])
        label = torch.tensor(self.labels[i])
        
        # Data augmentation (only during training)
        if self.training:
            # 1. Gaussian noise
            if torch.rand(1).item() < 0.5:
                eeg = eeg + torch.randn_like(eeg) * 0.05
            
            # 2. Channel dropout
            if torch.rand(1).item() < 0.3:
                n_drop = torch.randint(1, 6, (1,)).item()
                drop_idx = torch.randperm(eeg.size(0))[:n_drop]
                eeg[drop_idx] = 0
            
            # 3. Time shift
            if torch.rand(1).item() < 0.5:
                shift = torch.randint(-32, 33, (1,)).item()
                eeg = torch.roll(eeg, shift, dims=1)
                audio = torch.roll(audio, shift, dims=1)
        
        return eeg, audio, label
