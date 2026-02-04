# test_dtu_loader.py - Test DTU dataset loading

import glob
from clean_dtu import DTUDataset

# Find DTU .mat files
dtu_files = sorted(glob.glob('/home/durgesh/bhnu/data/dtu_data/S*_data_preproc.mat'))

print(f"Found {len(dtu_files)} DTU subject files\n")

if len(dtu_files) > 0:
    # Test loading first subject
    print("Testing DTU loader on first subject:")
    print("="*80)
    
    dataset = DTUDataset(
        subject_file=dtu_files[0],
        window_length=50.0,  # Your best window length
        fs_target=64,
        verbose=True
    )
    
    print(f"\n{'='*80}")
    print(f"Dataset loaded successfully!")
    print(f"Total windows: {len(dataset)}")
    print(f"EEG shape: {dataset.eeg_windows.shape}")
    print(f"Audio shape: {dataset.audio_windows.shape}")
    print(f"Labels shape: {dataset.labels.shape}")
    print(f"Label distribution: {dataset.labels.sum()} / {len(dataset.labels)} = {dataset.labels.mean():.2f}")
    print(f"{'='*80}\n")
    
    # Test getting a sample
    eeg, audio, label = dataset[0]
    print(f"Sample 0:")
    print(f"  EEG tensor: {eeg.shape} {eeg.dtype}")
    print(f"  Audio tensor: {audio.shape} {audio.dtype}")
    print(f"  Label: {label.item()}")
    
    print("\n✅ DTU loader working correctly!")
else:
    print("❌ No DTU files found!")
    print("Check path: /home/durgesh/bhnu/data/dtu_data/DATA_preproc/")
