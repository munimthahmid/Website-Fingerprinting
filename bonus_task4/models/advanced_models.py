"""
Bonus Task 4: Advanced Model Architectures
Contains ResNet, Transformer, and Ensemble models for beating 74.6% accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual block for ResNet-style architecture."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual
        out = self.relu(out)

        return out


class AdvancedResNetClassifier(nn.Module):
    """Advanced ResNet-style classifier with skip connections and attention."""

    def __init__(self, input_size, hidden_size, num_classes):
        super(AdvancedResNetClassifier, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 512), nn.Sigmoid()
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Reshape for 1D convolution
        x = x.unsqueeze(1)

        # Initial layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Attention
        attention_weights = self.attention(x)
        x = x * attention_weights

        # Classification
        x = self.classifier(x)

        return x


class TransformerClassifier(nn.Module):
    """Transformer-based classifier for sequential data."""

    def __init__(self, input_size, hidden_size, num_classes, num_heads=8, num_layers=4):
        super(TransformerClassifier, self).__init__()

        # Input projection
        self.input_projection = nn.Linear(1, hidden_size)

        # Positional encoding
        self.pos_encoding = self._get_positional_encoding(input_size, hidden_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def _get_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x):
        batch_size, seq_len = x.shape

        # Reshape and project
        x = x.unsqueeze(-1)  # (batch, seq, 1)
        x = self.input_projection(x)  # (batch, seq, hidden)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)

        # Transformer expects (seq, batch, hidden)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=0)  # Average over sequence length

        # Classification
        x = self.classifier(x)

        return x


class EnsembleClassifier(nn.Module):
    """Ensemble of multiple models for improved performance."""

    def __init__(self, models):
        super(EnsembleClassifier, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))

        # Average predictions
        ensemble_output = torch.stack(outputs).mean(dim=0)
        return ensemble_output


class AttentionCNN(nn.Module):
    """CNN with attention mechanism for better feature extraction."""

    def __init__(self, input_size, hidden_size, num_classes):
        super(AttentionCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Pooling
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Attention layers
        self.attention_conv = nn.Conv1d(512, 1, kernel_size=1)

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(512, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # Reshape for 1D convolution
        x = x.unsqueeze(1)

        # Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))

        # Attention mechanism
        attention_weights = torch.softmax(self.attention_conv(x), dim=2)
        x = x * attention_weights

        # Global pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        # Classification
        x = self.classifier(x)

        return x
