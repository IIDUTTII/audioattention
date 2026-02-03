# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception1D(nn.Module):
    def __init__(self, in_channels, out_channels_total, kernel_sizes=(3, 9, 15)):
        super().__init__()
        
        # Calculate channels per branch
        n_branches = 1 + len(kernel_sizes)
        assert out_channels_total % n_branches == 0
        b_ch = out_channels_total // n_branches
        
        # 1×1 branch
        self.branch_1x1 = nn.Conv1d(in_channels, b_ch, kernel_size=1)
        
        # Feature extraction branches
        self.branches_feat = nn.ModuleList()
        for k in kernel_sizes:
            pad = k // 2
            self.branches_feat.append(nn.Conv1d(in_channels, b_ch, kernel_size=k, padding=pad))
        
        self.bn = nn.BatchNorm1d(out_channels_total)
    
    def forward(self, x):
        # Process through all branches
        outs = [self.branch_1x1(x)]
        for conv in self.branches_feat:
            outs.append(conv(x))
        
        # Concatenate and normalize
        out = torch.cat(outs, dim=1)
        out = self.bn(F.relu(out))
        return out


class SimpleAADNet(nn.Module):
    def __init__(self, eeg_channels=64, audio_channels=1, feat_channels=64):  # ← Changed from 96 to 64
        super().__init__()
        
        # EEG encoder
        self.eeg_in = nn.BatchNorm1d(eeg_channels)
        self.eeg_block = Inception1D(eeg_channels, feat_channels, kernel_sizes=(19, 25, 33))
        self.eeg_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.eeg_bn2 = nn.BatchNorm1d(feat_channels)
        self.eeg_dropout = nn.Dropout(p=0.3)  # ← NEW: Add dropout after EEG encoder
        
        # Audio encoder
        self.audio_in = nn.BatchNorm1d(audio_channels)
        self.audio_block = Inception1D(audio_channels, feat_channels, kernel_sizes=(65, 81))
        self.audio_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.audio_bn2 = nn.BatchNorm1d(feat_channels)
        self.audio_dropout = nn.Dropout(p=0.3)  # ← NEW: Add dropout after audio encoder
        
        # Classifier
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2 * feat_channels, 2)
    
    def encode_eeg(self, eeg):
        x = self.eeg_in(eeg)
        x = self.eeg_block(x)
        x = self.eeg_pool(x)
        x = self.eeg_bn2(x)
        x = self.eeg_dropout(x)  # ← NEW: Apply dropout
        return x
    
    def encode_audio_single(self, audio):
        x = self.audio_in(audio)
        x = self.audio_block(x)
        x = self.audio_pool(x)
        x = self.audio_bn2(x)
        x = self.audio_dropout(x)  # ← NEW: Apply dropout
        return x
    
    @staticmethod
    def channel_corr(eeg_feat, audio_feat, eps=1e-8):
        """Compute channel-wise correlation"""
        eeg_mean = eeg_feat.mean(dim=2, keepdims=True)
        aud_mean = audio_feat.mean(dim=2, keepdims=True)
        eeg_c = eeg_feat - eeg_mean
        aud_c = audio_feat - aud_mean
        
        num = (eeg_c * aud_c).sum(dim=2)
        den = torch.sqrt((eeg_c ** 2).sum(dim=2) * (aud_c ** 2).sum(dim=2) + eps)
        corr = num / (den + eps)
        return corr
    
    def forward(self, eeg, audio):
        eeg_feat = self.encode_eeg(eeg)
        
        a0 = audio[:, 0:1, :]
        a1 = audio[:, 1:2, :]
        a0_feat = self.encode_audio_single(a0)
        a1_feat = self.encode_audio_single(a1)
        
        corr0 = self.channel_corr(eeg_feat, a0_feat)
        corr1 = self.channel_corr(eeg_feat, a1_feat)
        
        feats = torch.cat([corr0, corr1], dim=1)
        feats = self.dropout(feats)
        logits = self.fc(feats)
        return logits


# Test it
if __name__ == '__main__':
    model = SimpleAADNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    batch_size = 4
    eeg = torch.randn(batch_size, 64, 640)
    audio = torch.randn(batch_size, 2, 640)
    
    logits = model(eeg, audio)
    print(f"\nInput: EEG {eeg.shape}, Audio {audio.shape}")
    print(f"Output: Logits {logits.shape}")
    print(f"Predictions: {logits.argmax(dim=1)}")
