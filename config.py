# -*- coding: utf-8 -*-
"""
config.py
---------
Single source of truth for ALL experiment settings.

Why this matters for reproducibility:
  If you share this file with someone else, they can reproduce your exact
  results because every random decision (shuffling, weight initialisation,
  colour assignment) uses the same seed.
"""

# Reproducibility
SEED = 42           # Global random seed. Change to verify results aren't seed-dependent.

# Dataset settings
NUM_CLASSES  = 10   # MNIST has digits 0-9
DATASET_SIZE = 5000 # Training samples (small for speed -- full run < 30 min)
TEST_SIZE    = 1000 # Test samples

# Spurious-correlation strength.
# 0.0 = colour is random (no bias).  1.0 = colour perfectly predicts label.
# 0.9 means: 90% of the time, digit d gets its "assigned" colour.
CORRELATION  = 0.9

# Colour palette: each digit is assigned one "biased" colour (R, G, B) in 0-255.
DIGIT_COLORS = {
    0: (220,  50,  50),   # Red
    1: ( 50, 200,  50),   # Green
    2: ( 50, 100, 220),   # Blue
    3: (220, 180,  50),   # Yellow
    4: (180,  50, 220),   # Purple
    5: ( 50, 210, 210),   # Cyan
    6: (220, 120,  50),   # Orange
    7: (180, 220,  50),   # Lime
    8: (220,  50, 180),   # Pink
    9: (100,  50, 220),   # Violet
}

# Training settings (used in later steps)
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
EPOCHS        = 5   # Intentionally small so training finishes in minutes

# Output paths
FIGURES_DIR     = "outputs/figures"
CHECKPOINTS_DIR = "outputs/checkpoints"
