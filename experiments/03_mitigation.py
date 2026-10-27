# -*- coding: utf-8 -*-
"""
experiments/03_mitigation.py
-----------------------------
STEP 3 -- Mitigated CNN: train with grayscale augmentation.

Strategy: Grayscale Augmentation
---------------------------------
During TRAINING only, each image is randomly converted to grayscale
with probability p=0.5. This forces the network to build shape-based
features because on any batch ~half the images have no colour.

Evaluation conditions are IDENTICAL to the baseline:
  - Same architecture, same seed, same epochs, same test sets.
  - Test sets are never augmented.

How to run:
  python experiments/03_mitigation.py
"""

import os, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json

import config
from dataset.colored_mnist import ColoredMNIST, set_all_seeds
from models.cnn import BaselineCNN
from training.train import train
from training.evaluate import evaluate

# ---- Setup ---------------------------------------------------------------
print("=" * 65)
print("STEP 3: Mitigated CNN -- Grayscale Augmentation")
print(f"  Seed              : {config.SEED}")
print(f"  Augmentation prob : 0.5  (50% of images go grayscale)")
print(f"  Correlation       : {config.CORRELATION}")
print(f"  Epochs            : {config.EPOCHS}")
print("=" * 65)

set_all_seeds(config.SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device            : {device}")

os.makedirs(config.FIGURES_DIR,     exist_ok=True)
os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)

# ---- Augmentation --------------------------------------------------------
GRAYSCALE_PROB = 0.5

def grayscale_augment(img_tensor, prob, rng):
    """
    Randomly convert a (3, H, W) tensor in [0,1] to grayscale.
    Luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
    The result is broadcast back to 3 channels so the CNN input shape
    stays (3, H, W) with no architecture changes.
    """
    if rng.random() < prob:
        lum = (0.299 * img_tensor[0]
             + 0.587 * img_tensor[1]
             + 0.114 * img_tensor[2])
        img_tensor = lum.unsqueeze(0).expand(3, -1, -1)
    return img_tensor

aug_rng = np.random.default_rng(config.SEED)

def augmented_collate(batch):
    imgs, labels = zip(*batch)
    augmented = [grayscale_augment(img.clone(), GRAYSCALE_PROB, aug_rng)
                 for img in imgs]
    return torch.stack(augmented), torch.tensor(labels, dtype=torch.long)

# ---- Datasets ------------------------------------------------------------
print("\n[1/5] Building datasets ...")

train_ds = ColoredMNIST(
    train=True, num_samples=config.DATASET_SIZE,
    correlation=config.CORRELATION, seed=config.SEED,
)
biased_test_ds = ColoredMNIST(
    train=False, num_samples=config.TEST_SIZE,
    correlation=config.CORRELATION, seed=config.SEED + 1,
)
unbiased_test_ds = ColoredMNIST(
    train=False, num_samples=config.TEST_SIZE,
    correlation=0.0, seed=config.SEED + 99,
)

train_loader = DataLoader(
    train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
    num_workers=0, collate_fn=augmented_collate,
)
biased_test_loader = DataLoader(
    biased_test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0,
)
unbiased_test_loader = DataLoader(
    unbiased_test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0,
)

print(f"      Train samples  : {len(train_ds)}  (with 50% grayscale aug)")
print(f"      Biased test    : {len(biased_test_ds)}  (NO augmentation)")
print(f"      Unbiased test  : {len(unbiased_test_ds)} (NO augmentation)")

# ---- Model ---------------------------------------------------------------
print("\n[2/5] Initialising BaselineCNN (same architecture as baseline) ...")
model = BaselineCNN(num_classes=config.NUM_CLASSES).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"      Trainable parameters: {n_params:,}  (identical to baseline)")

# ---- Training ------------------------------------------------------------
print("\n[3/5] Training with grayscale augmentation ...")
t0      = time.time()
history = train(
    model=model,
    train_loader=train_loader,
    epochs=config.EPOCHS,
    lr=config.LEARNING_RATE,
    device=device,
    verbose=True,
)
elapsed = time.time() - t0
print(f"      Training complete in {elapsed:.1f}s")

ckpt_path = os.path.join(config.CHECKPOINTS_DIR, 'mitigated_cnn.pt')
torch.save(model.state_dict(), ckpt_path)
print(f"      Checkpoint saved -> {ckpt_path}")

# ---- Evaluation ----------------------------------------------------------
print("\n[4/5] Evaluating (SAME test sets as baseline) ...")
biased_results   = evaluate(model, biased_test_loader,   device, config.NUM_CLASSES)
unbiased_results = evaluate(model, unbiased_test_loader, device, config.NUM_CLASSES)
bias_score = biased_results['accuracy'] - unbiased_results['accuracy']

print()
print("  +" + "-" * 55 + "+")
print(f"  | {'Metric':<30} {'Biased':>10} {'Unbiased':>10} |")
print("  +" + "-" * 55 + "+")
print(f"  | {'Overall accuracy':<30} "
      f"{biased_results['accuracy']:>9.1%} "
      f"{unbiased_results['accuracy']:>9.1%} |")
print(f"  | {'Loss':<30} "
      f"{biased_results['loss']:>10.4f} "
      f"{unbiased_results['loss']:>10.4f} |")
print("  +" + "-" * 55 + "+")
print()
print(f"  Bias score (biased - unbiased accuracy): {bias_score:+.1%}")
if bias_score > 0.15:
    interp = "STRONG colour bias: augmentation had limited effect."
elif bias_score > 0.05:
    interp = "MODERATE colour bias: some improvement expected."
else:
    interp = "Low colour bias: grayscale augmentation appears effective."
print(f"  Interpretation: {interp}")
print()

COLOR_NAMES = ['Red','Green','Blue','Yellow','Purple',
               'Cyan','Orange','Lime','Pink','Violet']

print("  Per-class accuracy (biased test | unbiased test):")
print("  " + "-" * 52)
for d in range(config.NUM_CLASSES):
    ba = biased_results['per_class_acc'][d]
    ua = unbiased_results['per_class_acc'][d]
    print(f"  Digit {d} ({COLOR_NAMES[d]:>7}): biased={ba:.1%}  unbiased={ua:.1%}  drop={ba-ua:+.1%}")
print()

# ---- Figures -------------------------------------------------------------
print("[5/5] Saving figures ...")

epochs_range = [h['epoch'] for h in history]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(epochs_range, [h['loss'] for h in history], 'o-', color='#6db86d', linewidth=2)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Training Loss (with aug)'); ax1.grid(alpha=0.3)
ax2.plot(epochs_range, [h['accuracy']*100 for h in history], 'o-', color='#6d9eb8', linewidth=2)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training Accuracy (with aug)'); ax2.grid(alpha=0.3)
ax2.set_ylim(0, 100)
fig.suptitle('Mitigated CNN (Grayscale Aug) -- Training Curve', fontsize=12, fontweight='bold')
plt.tight_layout()
p = os.path.join(config.FIGURES_DIR, 'mitigated_training_curve.png')
plt.savefig(p, dpi=120, bbox_inches='tight'); plt.close()
print(f"      Saved -> {p}")

# Persist results for Step 4
results_path = os.path.join(config.CHECKPOINTS_DIR, 'mitigated_results.json')
with open(results_path, 'w') as f:
    json.dump({
        'biased_acc':         biased_results['accuracy'],
        'unbiased_acc':       unbiased_results['accuracy'],
        'bias_score':         bias_score,
        'per_class_biased':   biased_results['per_class_acc'].tolist(),
        'per_class_unbiased': unbiased_results['per_class_acc'].tolist(),
        'confusion_biased':   biased_results['confusion_matrix'].tolist(),
        'confusion_unbiased': unbiased_results['confusion_matrix'].tolist(),
    }, f, indent=2)
print(f"      Results saved -> {results_path}")

# ---- Summary -------------------------------------------------------------
print()
print("=" * 65)
print("STEP 3 SUMMARY")
print("=" * 65)
print(f"  Biased test accuracy  : {biased_results['accuracy']:.1%}")
print(f"  Unbiased test accuracy: {unbiased_results['accuracy']:.1%}")
print(f"  Bias score            : {bias_score:+.1%}")
print(f"  Interpretation        : {interp}")
print()
print("Next step: python experiments/04_comparison.py")
