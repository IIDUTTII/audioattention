# train.py (UPDATED VERSION)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import time
import matplotlib.pyplot as plt

from dataset import KULDataset
from model import SimpleAADNet


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    """One training epoch"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    use_amp = device.type == "cuda"
    
    for eeg, audio, labels in loader:
        eeg, audio, labels = eeg.to(device), audio.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):  # ← Fix deprecation warning
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
    """One validation epoch"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    use_amp = device.type == "cuda"
    
    with torch.no_grad():
        for eeg, audio, labels in loader:
            eeg, audio, labels = eeg.to(device), audio.to(device), labels.to(device)
            
            with autocast(enabled=use_amp):  # ← Fix deprecation warning
                logits = model(eeg, audio)
                loss = criterion(logits, labels)
            
            total_loss += loss.item() * eeg.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += eeg.size(0)
    
    return total_loss / total_samples, total_correct / total_samples


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    import glob,os
    from torch.utils.data import ConcatDataset
    # Load dataset
    subject_files = sorted(glob.glob('/home/durgesh/bhnu/data/kul_data/S*.mat'))
    print(f"Found {len(subject_files)} subject files")
    
    datasets = []
    for subject_file in subject_files:
        print(f"Loading {os.path.basename(subject_file)}...", end=" ")
        ds = KULDataset(
            subject_file=subject_file,
            stimuli_dir='/home/durgesh/bhnu/data/kul_data/stimuli',
            window_length=50.0,
            fs_target=64,
            training=False,
            verbose=False  # Don't print for each subject
        )
        datasets.append(ds)
        print(f"{len(ds)} windows")
    
    full_dataset = ConcatDataset(datasets)
    print(f"\n✅ Total dataset: {len(full_dataset)} windows from {len(subject_files)} subjects")
    # =======================================
    
    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    # Enable augmentation for training only
    train_ds.dataset.training = True
    val_ds.dataset.training = False
    print(f"Augmentation: Train=ON, Val=OFF")
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  # ← Reduced batch size
                              num_workers=8, pin_memory=True)  # ← Reduced workers
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, 
                           num_workers=8, pin_memory=True)
    
    print(f"\nDataset: {len(train_ds)} train, {len(val_ds)} val")
    
    # ========== UPDATED: Use 96 channels (divisible by 4) ==========
    model = SimpleAADNet(
        eeg_channels=64,
        audio_channels=1,
        feat_channels=48  # ← Back to 96 (4 branches × 24 = 96)
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters (feat_channels=96)")
    # ===============================================================
    
    criterion = nn.CrossEntropyLoss()
    
    # ========== FIXED: Gentler weight decay + learning rate ==========
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=5e-4,  # Standard learning rate
        weight_decay=5e-3  # ← Reduced from 5e-3 (was too strong)
    )
    print(f"Optimizer: Adam(lr=1e-3, weight_decay=1e-3)")
    # =================================================================
    
    scaler = GradScaler()  # ← Fix deprecation warning
    
    # ========== FIXED: Longer patience ==========
    n_epochs = 50
    best_val_acc = 0.0
    patience = 10  # ← Increased from 5 to 10
    patience_counter = 0
    # ===========================================
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print(f"\nTraining for up to {n_epochs} epochs (patience={patience})")
    print(f"Regularization: dropout=0.3/0.5, weight_decay=1e-3, augmentation=ON")
    print("-" * 80)
    
    for epoch in range(1, n_epochs + 1):
        start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        elapsed = time.time() - start
        
        gap = train_acc - val_acc
        print(f"Epoch {epoch:02d} | "
              f"Train: {train_loss:.4f}/{train_acc:.3f} | "
              f"Val: {val_loss:.4f}/{val_acc:.3f} | "
              f"Gap: {gap:+.3f} | "  # ← Show sign
              f"{elapsed:.1f}s")
        
        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  ✓ Best! Saved model")
        else:
            patience_counter += 1
            if patience_counter <= patience:  # Only show if within patience
                print(f"  No improvement ({patience_counter}/{patience})")
        
        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping at epoch {epoch}")
            print(f"   Best val acc was at epoch {epoch - patience}")
            break
    
    print("-" * 80)
    print(f"\n✅ Training complete!")
    print(f"   Best validation accuracy: {best_val_acc:.3f} ({best_val_acc*100:.1f}%)")
    
    # Calculate final overfitting gap
    if len(train_accs) > patience:
        final_train_acc = train_accs[max(0, len(train_accs) - patience - 1)]
        final_gap = final_train_acc - best_val_acc
        print(f"   Overfitting gap: {final_gap:.3f} ({final_gap*100:.1f}%)")
    
    # Plot results
    plot_curves(train_losses, train_accs, val_losses, val_accs)


def plot_curves(train_losses, train_accs, val_losses, val_accs):
    """Plot training curves"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    axes[1].axhline(y=0.5, color='gray', linestyle='--', label='Chance (50%)', alpha=0.5)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.4, 1.0])  # ← Better y-axis range
    
    plt.tight_layout()
    plt.savefig('training_curves_improved2.png', dpi=150)
    print(f"\n📊 Saved: training_curves_improved2.png")


if __name__ == '__main__':
    main()
