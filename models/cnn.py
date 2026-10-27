# -*- coding: utf-8 -*-
"""
models/cnn.py
-------------
Baseline CNN for Colored MNIST.

Architecture
------------
Input  : (batch, 3, 28, 28)  -- 3-channel RGB image, 28x28 pixels

Block 1: Conv(3->32, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
          Output: (batch, 32, 13, 13)

Block 2: Conv(32->64, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
          Output: (batch, 64, 5, 5)

Flatten -> FC(64*5*5 -> 256) -> ReLU -> Dropout(0.4)
        -> FC(256 -> 10)      -- 10 logit scores, one per digit

Why this architecture?
  - Tiny enough to train in ~2 minutes on CPU.
  - Deep enough to learn colour shortcuts when they exist.
  - BatchNorm stabilises training; Dropout prevents over-reliance on
    any single neuron.
  - No sigmoid/softmax here -- CrossEntropyLoss expects raw logits.
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()

        # ----- Convolutional blocks -----
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # (3,28,28)->(32,28,28)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # (32,14,14)

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (32,14,14)->(64,14,14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # (64,7,7)
        )

        # ----- Fully-connected head -----
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, 3, 28, 28)
        returns : (batch, num_classes)  -- raw logits
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
