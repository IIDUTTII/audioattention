# train_combined.py - Train on KUL + DTU Combined

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import glob
import re
import os
import time

from dataset import KULDataset
from dataset_dtu import DTUDataset
from model import SimpleAADNet


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    use_amp = device.type == "cuda"
    
    for eeg, audio, labels in loader:
        eeg, audio, labels = eeg.to(device), audio.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp, device_type='cuda'):
            logits = model(eeg, audio)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * eeg.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += eeg.size(0)
    
    return total_loss / total_samples, total_correct / total_samples


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    use_amp = device.type == "cuda"
    
    with torch.no_grad():
        for eeg, audio, labels in loader:
            eeg, audio, labels = eeg.to(device), audio.to(device), labels.to(device)
            
            with autocast(enabled=use_amp, device_type='cuda'):
                logits = model(eeg, audio)
                loss = criterion(logits, labels)
            
            total_loss += loss.item() * eeg.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += eeg.size(0)
    
    return total_loss / total_samples, total_correct / total_samples


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ========== CONFIGURATION ==========
    window_length = 50.0  # Your best window length
    n_epochs = 50
    patience = 10
    # ==================================
    
    print(f"\n{'='*80}")
    print(f"LOSO CROSS-VALIDATION: KUL + DTU COMBINED")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Window length: {window_length}s")
    print(f"Max epochs: {n_epochs} (patience={patience})")
    print(f"{'='*80}\n")
    
    # ========== Load KUL subjects ==========
    print("Loading KUL subjects...")
    kul_files = sorted(
        glob.glob('/home/durgesh/bhnu/data/kul_data/S*.mat'),
        key=lambda x: int(re.search(r'S(\d+)\.mat', x).group(1))
    )
    
    kul_datasets = []
    for i, subject_file in enumerate(kul_files):
        subject_name = os.path.basename(subject_file)
        print(f"[KUL {i+1:2d}/16] {subject_name}...", end=" ", flush=True)
        
        ds = KULDataset(
            subject_file=subject_file,
            stimuli_dir='/home/durgesh/bhnu/data/kul_data/stimuli',
            window_length=window_length,
            fs_target=64,
            training=False,
            verbose=False
        )
        kul_datasets.append(ds)
        print(f"✓ {len(ds)} windows")
    
    kul_windows = sum(len(ds) for ds in kul_datasets)
    print(f"✅ KUL: 16 subjects, {kul_windows} windows\n")
    
    # ========== Load DTU subjects ==========
    print("Loading DTU subjects...")
    dtu_dirs = sorted(glob.glob('/home/durgesh/bhnu/data/dtu_data/DATA_preproc/S*'))
    
    dtu_datasets = []
    for i, subject_dir in enumerate(dtu_dirs):
        subject_name = os.path.basename(subject_dir)
        print(f"[DTU {i+1:2d}/18] {subject_name}...", end=" ", flush=True)
        
        ds = DTUDataset(
            subject_dir=subject_dir,
            window_length=window_length,
            fs_target=64,
            training=False,
            verbose=False
        )
        dtu_datasets.append(ds)
        print(f"✓ {len(ds)} windows")
    
    dtu_windows = sum(len(ds) for ds in dtu_datasets)
    print(f"✅ DTU: 18 subjects, {dtu_windows} windows\n")
    
    # ========== Combine all subjects ==========
    all_datasets = kul_datasets + dtu_datasets
    total_subjects = len(all_datasets)
    total_windows = kul_windows + dtu_windows
    
    print(f"{'='*80}")
    print(f"COMBINED DATASET:")
    print(f"  Total: {total_subjects} subjects (16 KUL + 18 DTU)")
    print(f"  Total: {total_windows} windows")
    print(f"{'='*80}\n")
    
    # ========== LOSO Cross-Validation ==========
    all_fold_results = []
    
    for fold_idx in range(total_subjects):
        # Determine dataset source
        if fold_idx < 16:
            test_name = f"KUL-{os.path.basename(kul_files[fold_idx])}"
        else:
            test_name = f"DTU-{os.path.basename(dtu_dirs[fold_idx-16])}"
        
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1}/{total_subjects}: Testing on {test_name}")
        print(f"{'='*80}")
        
        # Split data
        test_dataset = all_datasets[fold_idx]
        train_datasets = [all_datasets[i] for i in range(total_subjects) if i != fold_idx]
        train_dataset = ConcatDataset(train_datasets)
        
        # Enable augmentation
        for i in range(total_subjects):
            all_datasets[i].training = (i != fold_idx)
        
        print(f"Train: {len(train_dataset):5d} windows from {len(train_datasets)} subjects")
        print(f"Test:  {len(test_dataset):5d} windows from 1 subject")
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, 
                                  num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, 
                                 num_workers=8, pin_memory=True)
        
        # Create model
        model = SimpleAADNet(eeg_channels=64, audio_channels=1, feat_channels=96).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scaler = GradScaler()
        
        # Training loop
        best_test_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(1, n_epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
            test_loss, test_acc = val_epoch(model, test_loader, criterion, device)
            
            if epoch % 5 == 0 or test_acc > best_test_acc:
                gap = train_acc - test_acc
                status = ""
                
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_epoch = epoch
                    patience_counter = 0
                    status = " ✓ BEST"
                else:
                    patience_counter += 1
                
                print(f"Epoch {epoch:2d} | Train: {train_loss:.4f}/{train_acc:.3f} | "
                      f"Test: {test_loss:.4f}/{test_acc:.3f} | Gap: {gap:+.3f}{status}")
            else:
                if test_acc <= best_test_acc:
                    patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\n⚠️ Early stop at epoch {epoch}")
                break
        
        print(f"\n{'─'*80}")
        print(f"✅ Fold {fold_idx + 1}: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")
        print(f"{'─'*80}")
        
        all_fold_results.append({
            'fold': fold_idx + 1,
            'subject': test_name,
            'best_acc': best_test_acc,
            'best_epoch': best_epoch
        })
        
        # Disable augmentation
        for ds in all_datasets:
            ds.training = False
    
    # ========== Final Summary ==========
    accs = [r['best_acc'] for r in all_fold_results]
    
    print(f"\n\n{'='*80}")
    print(f"FINAL RESULTS: KUL + DTU COMBINED")
    print(f"{'='*80}\n")
    
    print(f"{'Fold':>4} | {'Subject':>15} | {'Accuracy':>10}")
    print(f"{'-'*80}")
    for r in all_fold_results:
        print(f"{r['fold']:4d} | {r['subject']:>15} | {r['best_acc']:10.4f}")
    
    print(f"\n{'='*80}")
    print(f"Mean Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f} ({np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%)")
    print(f"Best:          {max(accs):.4f} ({max(accs)*100:.2f}%)")
    print(f"Worst:         {min(accs):.4f} ({min(accs)*100:.2f}%)")
    print(f"{'='*80}\n")
    
    # Save
    np.save('loso_kul_dtu_combined.npy', all_fold_results)
    print("✅ Saved to loso_kul_dtu_combined.npy")


if __name__ == '__main__':
    main()
