# -*- coding: utf-8 -*-
"""
experiments/02_baseline_cnn.py
--------------------------------
STEP 2 -- Train the biased baseline CNN and quantify colour reliance.

What this script does
---------------------
1. Builds TWO test sets from the same raw MNIST images:
     - biased_test   (correlation=0.9): colours match training distribution
     - unbiased_test (correlation=0.0): colours are completely random

2. Trains BaselineCNN on the biased training set.

3. Evaluates on BOTH test sets. The key metric is:

     bias_score = biased_accuracy - unbiased_accuracy

   A large bias_score (e.g. >20%) means the model learned colour as a
   shortcut. A small score means it learned real digit shapes.
   We do NOT claim a result until we have actually run the numbers.

4. Saves:
     outputs/checkpoints/baseline_cnn.pt   -- trained model weights
     outputs/figures/training_curve.png    -- loss + accuracy over epochs
     outputs/figures/per_class_acc.png     -- per-digit accuracy (biased vs unbiased)
     outputs/figures/confusion_biased.png  -- confusion matrix, biased test
     outputs/figures/confusion_unbiased.png -- confusion matrix, unbiased test

How to run
----------
  python experiments/02_baseline_cnn.py

Expected runtime: 3-8 minutes on CPU.
"""

import os, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from dataset.colored_mnist import ColoredMNIST, set_all_seeds, get_dataloaders
from models.cnn import BaselineCNN
from training.train import train
from training.evaluate import evaluate

# ------------------------------------------------------------------ Setup
print("=" * 65)
print("STEP 2: Baseline CNN training")
print(f"  Seed         : {config.SEED}")
print(f"  Correlation  : {config.CORRELATION}")
print(f"  Epochs       : {config.EPOCHS}")
print(f"  Batch size   : {config.BATCH_SIZE}")
print(f"  Learning rate: {config.LEARNING_RATE}")
print("=" * 65)

set_all_seeds(config.SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device       : {device}")

os.makedirs(config.FIGURES_DIR,     exist_ok=True)
os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)

# ------------------------------------------------------------------ Datasets
print("\n[1/5] Building datasets ...")

# Training set: biased (correlation = 0.9)
train_loader, _ = get_dataloaders(
    correlation=config.CORRELATION,
    seed=config.SEED,
)

# Biased test set: same distribution as training
_, biased_test_loader = get_dataloaders(
    correlation=config.CORRELATION,
    seed=config.SEED,
)

# Unbiased test set: correlation = 0.0 (colours are random)
# This is how we measure whether the model ACTUALLY learned shapes.
unbiased_test_ds = ColoredMNIST(
    train=False,
    num_samples=config.TEST_SIZE,
    correlation=0.0,       # <-- key difference
    seed=config.SEED + 99,
)
from torch.utils.data import DataLoader
unbiased_test_loader = DataLoader(
    unbiased_test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
)

print(f"      Train samples  : {len(train_loader.dataset)}")
print(f"      Biased test    : {len(biased_test_loader.dataset)}  (correlation={config.CORRELATION})")
print(f"      Unbiased test  : {len(unbiased_test_loader.dataset)} (correlation=0.0)")

# ------------------------------------------------------------------ Model
print("\n[2/5] Initialising BaselineCNN ...")
model = BaselineCNN(num_classes=config.NUM_CLASSES).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"      Trainable parameters: {n_params:,}")

# ------------------------------------------------------------------ Training
print("\n[3/5] Training ...")
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

# Save weights
ckpt_path = os.path.join(config.CHECKPOINTS_DIR, "baseline_cnn.pt")
torch.save(model.state_dict(), ckpt_path)
print(f"      Checkpoint saved -> {ckpt_path}")

# ------------------------------------------------------------------ Evaluation
print("\n[4/5] Evaluating ...")
biased_results   = evaluate(model, biased_test_loader,   device, config.NUM_CLASSES)
unbiased_results = evaluate(model, unbiased_test_loader, device, config.NUM_CLASSES)

bias_score = biased_results["accuracy"] - unbiased_results["accuracy"]

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
    interp = "STRONG colour bias: the model is a shortcut learner."
elif bias_score > 0.05:
    interp = "MODERATE colour bias detected."
else:
    interp = "Low colour bias: the model learned mostly from shape."
print(f"  Interpretation: {interp}")
print()

COLOR_NAMES = ["Red","Green","Blue","Yellow","Purple",
               "Cyan","Orange","Lime","Pink","Violet"]

print("  Per-class accuracy (biased test | unbiased test):")
print("  " + "-" * 52)
for d in range(config.NUM_CLASSES):
    ba = biased_results["per_class_acc"][d]
    ua = unbiased_results["per_class_acc"][d]
    print(f"  Digit {d} ({COLOR_NAMES[d]:>7}): biased={ba:.1%}  unbiased={ua:.1%}  drop={ba-ua:+.1%}")
print()

# ------------------------------------------------------------------ Figures
print("[5/5] Saving figures ...")

# --- Figure A: Training curve ---
epochs_range = [h["epoch"] for h in history]
losses       = [h["loss"]     for h in history]
accs         = [h["accuracy"] for h in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(epochs_range, losses, "o-", color="#e05c5c", linewidth=2)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-entropy loss")
ax1.set_title("Training Loss"); ax1.grid(alpha=0.3)

ax2.plot(epochs_range, [a * 100 for a in accs], "o-", color="#5ca8e0", linewidth=2)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training Accuracy"); ax2.grid(alpha=0.3)
ax2.set_ylim(0, 100)

fig.suptitle("Baseline CNN -- Training Curve", fontsize=12, fontweight="bold")
plt.tight_layout()
p = os.path.join(config.FIGURES_DIR, "training_curve.png")
plt.savefig(p, dpi=120, bbox_inches="tight"); plt.close()
print(f"      Saved -> {p}")

# --- Figure B: Per-class accuracy bar chart ---
x     = np.arange(config.NUM_CLASSES)
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5))
bars1 = ax.bar(x - width/2, biased_results["per_class_acc"]   * 100,
               width, label=f"Biased test   ({biased_results['accuracy']:.1%} overall)",
               color="#5ca8e0", alpha=0.85)
bars2 = ax.bar(x + width/2, unbiased_results["per_class_acc"] * 100,
               width, label=f"Unbiased test ({unbiased_results['accuracy']:.1%} overall)",
               color="#e07d5c", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([f"{d}\n{COLOR_NAMES[d]}" for d in range(10)], fontsize=9)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 105)
ax.set_title(
    f"Per-class Accuracy -- Baseline CNN\n"
    f"Bias score (biased - unbiased) = {bias_score:+.1%}",
    fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.axhline(90, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(9.6, 90.5, "90%", fontsize=8, color="gray")

plt.tight_layout()
p = os.path.join(config.FIGURES_DIR, "per_class_acc.png")
plt.savefig(p, dpi=120, bbox_inches="tight"); plt.close()
print(f"      Saved -> {p}")

# --- Figure C+D: Confusion matrices ---
def plot_confusion(conf_mat, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    # Normalise rows to fractions
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    frac     = conf_mat / (row_sums + 1e-8)
    im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(range(10)); ax.set_yticklabels(range(10))
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label",      fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for r in range(10):
        for c in range(10):
            v  = frac[r, c]
            tc = "white" if v > 0.5 else "black"
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7, color=tc)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()

plot_confusion(
    biased_results["confusion_matrix"],
    f"Confusion Matrix -- Biased test  (acc={biased_results['accuracy']:.1%})",
    os.path.join(config.FIGURES_DIR, "confusion_biased.png"),
)
print(f"      Saved -> {os.path.join(config.FIGURES_DIR, 'confusion_biased.png')}")

plot_confusion(
    unbiased_results["confusion_matrix"],
    f"Confusion Matrix -- Unbiased test  (acc={unbiased_results['accuracy']:.1%})",
    os.path.join(config.FIGURES_DIR, "confusion_unbiased.png"),
)
print(f"      Saved -> {os.path.join(config.FIGURES_DIR, 'confusion_unbiased.png')}")

# ------------------------------------------------------------------ Summary
print()
print("=" * 65)
print("STEP 2 SUMMARY")
print("=" * 65)
print(f"  Biased test accuracy  : {biased_results['accuracy']:.1%}")
print(f"  Unbiased test accuracy: {unbiased_results['accuracy']:.1%}")
print(f"  Bias score            : {bias_score:+.1%}")
print(f"  Interpretation        : {interp}")
print()
print("  Figures saved to outputs/figures/")
print("  Model saved to", ckpt_path)
print()
print("Next step: python experiments/03_mitigation.py")
