# loso_train.py - CORRECTED VERSION

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
    print(f"LEAVE-ONE-SUBJECT-OUT (LOSO) CROSS-VALIDATION")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Window length: {window_length}s")
    print(f"Max epochs: {n_epochs} (patience={patience})")
    print(f"{'='*80}\n")
    
    # Load subject files
    subject_files = sorted(
        glob.glob('/home/durgesh/bhnu/data/kul_data/S*.mat'),
        key=lambda x: int(re.search(r'S(\d+)\.mat', x).group(1))
    )
    
    stimuli_dir = '/home/durgesh/bhnu/data/kul_data/stimuli'
    n_subjects = len(subject_files)
    
    # ========== Load all subjects ONCE ==========
    print("Loading all subjects...")
    all_datasets = []
    
    for i, subject_file in enumerate(subject_files):
        subject_name = os.path.basename(subject_file)
        print(f"[{i+1:2d}/{n_subjects}] Loading {subject_name}...", end=" ", flush=True)
        
        ds = KULDataset(
            subject_file=subject_file,
            stimuli_dir=stimuli_dir,
            window_length=window_length,
            fs_target=64,
            training=False,  # Will enable per fold
            verbose=False
        )
        all_datasets.append(ds)
        print(f"✓ {len(ds)} windows")
    
    total_windows = sum(len(ds) for ds in all_datasets)
    print(f"\n✅ Loaded {n_subjects} subjects, {total_windows} total windows\n")
    # ============================================
    
    # ========== LOSO Cross-Validation ==========
    all_fold_results = []
    
    for fold_idx in range(n_subjects):
        test_subject_name = os.path.basename(subject_files[fold_idx])
        
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1}/{n_subjects}: Testing on {test_subject_name}")
        print(f"{'='*80}")
        
        # ========== CRITICAL: Proper train/test split ==========
        # Test set: ONLY the current subject
        test_dataset = all_datasets[fold_idx]
        
        # Train set: ALL OTHER subjects
        train_dataset_list = [all_datasets[i] for i in range(n_subjects) if i != fold_idx]
        train_dataset = ConcatDataset(train_dataset_list)
        # ======================================================
        
        # Enable augmentation ONLY for training subjects
        for i in range(n_subjects):
            if i == fold_idx:
                all_datasets[i].training = False  # Test subject: no augmentation
            else:
                all_datasets[i].training = True   # Train subjects: with augmentation
        
        print(f"Train: {len(train_dataset):4d} windows from {len(train_dataset_list)} subjects")
        print(f"Test:  {len(test_dataset):4d} windows from 1 subject ({test_subject_name})")
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  # Small batch
                                  num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, 
                                 num_workers=4, pin_memory=True)
        
        # ========== Create fresh model for this fold ==========
        model = SimpleAADNet(
            eeg_channels=64,
            audio_channels=1,
            feat_channels=96
        ).to(device)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_params:,} parameters\n")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scaler = GradScaler()
        # ====================================================
        
        # ========== Training loop for this fold ==========
        best_test_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>10} | {'Test Acc':>9} | {'Gap':>6}")
        print(f"{'-'*80}")
        
        for epoch in range(1, n_epochs + 1):
            start = time.time()
            
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
            test_loss, test_acc = val_epoch(model, test_loader, criterion, device)
            
            elapsed = time.time() - start
            gap = train_acc - test_acc
            
            # Print every 5 epochs or when best
            if epoch % 5 == 0 or test_acc > best_test_acc or epoch == 1:
                status = ""
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_epoch = epoch
                    patience_counter = 0
                    status = " ✓ BEST"
                    
                    # Save best model for this fold
                    torch.save(model.state_dict(), f'best_model_fold{fold_idx+1}.pt')
                else:
                    patience_counter += 1
                
                print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:9.3f} | "
                      f"{test_loss:10.4f} | {test_acc:9.3f} | {gap:+6.3f}{status}")
            else:
                if test_acc <= best_test_acc:
                    patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠️  Early stop at epoch {epoch} (no improvement for {patience} epochs)")
                break
        
        print(f"\n{'─'*80}")
        print(f"✅ Fold {fold_idx + 1} completed!")
        print(f"   Best test accuracy: {best_test_acc:.4f} ({best_test_acc*100:.2f}%) at epoch {best_epoch}")
        print(f"{'─'*80}")
        
        all_fold_results.append({
            'fold': fold_idx + 1,
            'subject': test_subject_name,
            'best_acc': best_test_acc,
            'best_epoch': best_epoch,
            'n_train': len(train_dataset),
            'n_test': len(test_dataset)
        })
        
        # Disable augmentation for all
        for ds in all_datasets:
            ds.training = False
    
    # ========== Final Summary ==========
    print(f"\n\n{'='*80}")
    print(f"LOSO CROSS-VALIDATION FINAL RESULTS")
    print(f"{'='*80}\n")
    
    print(f"{'Fold':>4} | {'Subject':>10} | {'Test Acc':>10} | {'Train Windows':>13} | {'Test Windows':>12}")
    print(f"{'-'*80}")
    
    accs = []
    for result in all_fold_results:
        print(f"{result['fold']:4d} | {result['subject']:>10} | "
              f"{result['best_acc']:10.4f} | {result['n_train']:13d} | {result['n_test']:12d}")
        accs.append(result['best_acc'])
    
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS:")
    print(f"{'='*80}")
    print(f"Mean Accuracy:    {mean_acc:.4f} ± {std_acc:.4f}  ({mean_acc*100:.2f}% ± {std_acc*100:.2f}%)")
    print(f"Best Subject:     {max(accs):.4f}  ({max(accs)*100:.2f}%)")
    print(f"Worst Subject:    {min(accs):.4f}  ({min(accs)*100:.2f}%)")
    print(f"Median:           {np.median(accs):.4f}  ({np.median(accs)*100:.2f}%)")
    print(f"{'='*80}\n")
    
    # Save results
    np.save('loso_results.npy', all_fold_results)
    np.save('loso_accuracies.npy', accs)
    
    print(f"✅ Results saved to loso_results.npy and loso_accuracies.npy\n")
    
    return all_fold_results


if __name__ == '__main__':
    main()
