import torch
import torch.nn as nn

class AudioClassifier(nn.Module):
    def __init__(self,
                 in_channels=1,
                 conv_channels=64,
                 embed_dim=256,
                 num_heads=4,
                 num_layers=4,
                 num_classes=10):
        super().__init__()

        # 1. Convolutional Downsampling (Reduce Freq and Time)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, kernel_size=3, stride=2, padding=1),  # (B, 64, 32, 64)
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),  # (B, 64, 16, 32)
            nn.BatchNorm2d(conv_channels),
            nn.ReLU()
        )

        # 2. Flatten and Linear Projection
        # Each time frame becomes a token
        self.proj = nn.Linear(conv_channels * 16, embed_dim)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        """
        x: (B, 1, 64, 128) — [Batch, Channels, Freq, Time]
        """
        x = self.conv(x)  # (B, 64, 16, 32)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)  # (B, T=32, C*F=64*16=1024)
        x = self.proj(x)  # (B, T=32, embed_dim)
        x = self.transformer(x)  # (B, T=32, embed_dim)
        x = x.mean(dim=1)  # Global average pooling over time
        return self.classifier(x)  # (B, num_classes)
