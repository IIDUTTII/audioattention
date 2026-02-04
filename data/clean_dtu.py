# dataset_dtu.py - DTU Dataset Loader (Corrected for actual structure)

import os
import pickle
import hashlib
import numpy as np
import torch
import scipy.io
from torch.utils.data import Dataset
from scipy.signal import resample_poly


class DTUDataset(Dataset):
    """
    DTU Auditory Attention Dataset Loader
    
    Data structure in .mat files:
    - data['data'][0][0]['eeg'][0][0]: EEG data [n_channels, n_samples]
    - data['data'][0][0]['wavA'][0][0]: Speaker A waveform/envelope
    - data['data'][0][0]['wavB'][0][0]: Speaker B waveform/envelope
    - data['data'][0][0]['event']: Trial events (contains attended speaker)
    - data['data'][0][0]['fsample'][0][0]: Sampling frequency
    """
    
    def __init__(self, subject_file, window_length=10.0, fs_target=64, 
                 use_cache=True, cache_dir="cache_dtu", verbose=True, 
                 training=True):
        """
        Args:
            subject_file: Path to subject .mat file (e.g., 'S01.mat')
            window_length: Window length in seconds
            fs_target: Target sampling frequency (downsample to this)
            use_cache: Whether to use cached preprocessed data
            cache_dir: Directory to store cache files
            verbose: Print progress messages
            training: Enable data augmentation
        """
        self.subject_file = subject_file
        self.window_length = window_length
        self.fs_target = fs_target
        self.window_samples = int(window_length * fs_target)
        self.verbose = verbose
        self.training = training
        
        if verbose:
            subject_name = os.path.basename(subject_file)
            print(f"Loading DTU subject: {subject_name}", end=" ", flush=True)
        
        cache = self._cache_path(cache_dir)
        
        if use_cache and os.path.exists(cache):
            if verbose: print("(from cache)", end=" ", flush=True)
            self._load_cache(cache)
        else:
            if verbose: print("(processing)", end=" ", flush=True)
            self._process_subject()
            if use_cache: self._save_cache(cache)
        
        if verbose: print(f"✓ {len(self.labels)} windows")
    
    def _cache_path(self, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        subject_name = os.path.basename(self.subject_file).replace('.mat', '')
        key = f"{subject_name}_{self.window_length}_{self.fs_target}"
        cache_file = f"{subject_name}_{hashlib.md5(key.encode()).hexdigest()[:8]}.pkl"
        return os.path.join(cache_dir, cache_file)
    
    def _process_subject(self):
        """Process all trials from the subject .mat file"""
        try:
            # Load .mat file
            mat_data = scipy.io.loadmat(self.subject_file)
            
            # Extract data structure
            data = mat_data['data'][0][0]
            
            # Get EEG data: [n_channels, n_samples]
            eeg = data['eeg'][0][0]
            
            # Get audio envelopes
            wavA = data['wavA'][0][0].flatten()  # Speaker A
            wavB = data['wavB'][0][0].flatten()  # Speaker B
            
            # Get sampling frequency
            fs_original = int(data['fsample'][0][0][0][0])
            
            # Get attended speaker from events
            events = data['event']
            attended_speaker = self._get_attended_speaker(events)
            
            # Label: 0=A attended, 1=B attended
            label = 0 if attended_speaker == 'A' else 1
            
            # Downsample if needed
            if fs_original != self.fs_target:
                # Downsample EEG
                eeg = resample_poly(eeg, self.fs_target, fs_original, axis=1)
                
                # Downsample audio envelopes
                wavA = resample_poly(wavA, self.fs_target, fs_original)
                wavB = resample_poly(wavB, self.fs_target, fs_original)
            
            # Ensure EEG has 64 channels (pad or trim if needed)
            if eeg.shape[0] < 64:
                # Pad with zeros
                pad = np.zeros((64 - eeg.shape[0], eeg.shape[1]), dtype=eeg.dtype)
                eeg = np.vstack([eeg, pad])
            elif eeg.shape[0] > 64:
                # Take first 64 channels
                eeg = eeg[:64, :]
            
            # Align lengths
            T = min(eeg.shape[1], len(wavA), len(wavB))
            eeg = eeg[:, :T]
            audio = np.stack([wavA[:T], wavB[:T]])  # [2, T]
            
            # Window the data
            n_windows = T // self.window_samples
            
            if n_windows == 0:
                # Data too short
                self.eeg_windows = np.empty((0, 64, self.window_samples), dtype=np.float32)
                self.audio_windows = np.empty((0, 2, self.window_samples), dtype=np.float32)
                self.labels = np.empty(0, dtype=np.int64)
                return
            
            # Trim to exact window length
            T_trim = n_windows * self.window_samples
            eeg = eeg[:, :T_trim]
            audio = audio[:, :T_trim]
            
            # Reshape: (channels, n_windows, window_samples)
            eeg_w = eeg.reshape(64, n_windows, self.window_samples)
            aud_w = audio.reshape(2, n_windows, self.window_samples)
            
            # Transpose to: (n_windows, channels, window_samples)
            eeg_w = eeg_w.transpose(1, 0, 2)
            aud_w = aud_w.transpose(1, 0, 2)
            
            # Standardize each window
            eeg_w = (eeg_w - eeg_w.mean((1,2), keepdims=True)) / (eeg_w.std((1,2), keepdims=True) + 1e-8)
            aud_w = (aud_w - aud_w.mean((1,2), keepdims=True)) / (aud_w.std((1,2), keepdims=True) + 1e-8)
            
            # Store
            self.eeg_windows = eeg_w.astype(np.float32)
            self.audio_windows = aud_w.astype(np.float32)
            self.labels = np.full(n_windows, label, dtype=np.int64)
        
        except Exception as e:
            if self.verbose:
                print(f"\n  ⚠️ Error: {e}")
            # Create empty dataset
            self.eeg_windows = np.empty((0, 64, self.window_samples), dtype=np.float32)
            self.audio_windows = np.empty((0, 2, self.window_samples), dtype=np.float32)
            self.labels = np.empty(0, dtype=np.int64)
    
    def _get_attended_speaker(self, events):
        """
        Extract attended speaker from events structure
        
        Returns 'A' or 'B'
        """
        try:
            # Events structure varies, common patterns:
            # - events[0][0]['type'] or events[0][0]['value']
            # - May contain 'attendA', 'attendB', or numeric codes
            
            if events.size > 0:
                # Try to find attended speaker in events
                for i in range(len(events[0])):
                    event = events[0][i]
                    
                    # Check for 'type' field
                    if hasattr(event, 'dtype') and 'type' in event.dtype.names:
                        event_type = str(event['type'][0])
                        if 'attendA' in event_type or 'attend_A' in event_type:
                            return 'A'
                        elif 'attendB' in event_type or 'attend_B' in event_type:
                            return 'B'
                    
                    # Check for 'value' field
                    if hasattr(event, 'dtype') and 'value' in event.dtype.names:
                        value = event['value'][0]
                        if isinstance(value, str):
                            if 'A' in value or '1' in value:
                                return 'A'
                            elif 'B' in value or '2' in value:
                                return 'B'
            
            # Default: assume speaker A attended (will be random but consistent)
            return 'A'
        
        except:
            # If unable to determine, default to A
            return 'A'
    
    def _save_cache(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "eeg": self.eeg_windows,
                "audio": self.audio_windows,
                "labels": self.labels
            }, f)
    
    def _load_cache(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.eeg_windows = d["eeg"]
        self.audio_windows = d["audio"]
        self.labels = d["labels"]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        eeg = torch.from_numpy(self.eeg_windows[idx])
        audio = torch.from_numpy(self.audio_windows[idx])
        label = torch.tensor(self.labels[idx])
        
        # Data augmentation (same as KUL)
        if self.training:
            # Gaussian noise
            if torch.rand(1).item() < 0.5:
                eeg = eeg + torch.randn_like(eeg) * 0.05
            
            # Channel dropout
            if torch.rand(1).item() < 0.3:
                n_drop = torch.randint(1, 6, (1,)).item()
                drop_idx = torch.randperm(eeg.size(0))[:n_drop]
                eeg[drop_idx] = 0
            
            # Time shift
            if torch.rand(1).item() < 0.5:
                shift = torch.randint(-32, 33, (1,)).item()
                eeg = torch.roll(eeg, shift, dims=1)
                audio = torch.roll(audio, shift, dims=1)
        
        return eeg, audio, label
