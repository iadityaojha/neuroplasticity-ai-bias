# -*- coding: utf-8 -*-
"""
experiments/01_dataset_check.py
--------------------------------
STEP 1 -- Generate the Colored MNIST dataset and visually verify
          that the spurious colour-label correlation is present.

This script produces:
  outputs/figures/sample_grid.png
    A 10x8 grid of coloured digits. Each row = one digit (0-9).
    Digits in the same row should share a dominant colour.

  outputs/figures/colour_distribution.png
    A heatmap (rows=digits, columns=colours).
    The diagonal should be approximately 0.90.

How to run:
  python experiments/01_dataset_check.py
"""

import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from dataset.colored_mnist import ColoredMNIST, set_all_seeds

# ------------------------------------------------------------------ Setup
print("=" * 60)
print("STEP 1: Dataset generation and visual verification")
print(f"  Random seed  : {config.SEED}")
print(f"  Correlation  : {config.CORRELATION}")
print(f"  Dataset size : {config.DATASET_SIZE}")
print("=" * 60)

set_all_seeds(config.SEED)
os.makedirs(config.FIGURES_DIR, exist_ok=True)

# ------------------------------------------------------------------ Build dataset
print("\n[1/4] Building ColoredMNIST dataset ...")
dataset = ColoredMNIST(
    train=True,
    num_samples=config.DATASET_SIZE,
    correlation=config.CORRELATION,
    seed=config.SEED,
)
print(f"      {len(dataset)} samples ready.")

# ------------------------------------------------------------------ Figure 1: Sample grid
print("\n[2/4] Creating sample grid (Figure 1) ...")

COLS = 8   # examples per digit
fig, axes = plt.subplots(config.NUM_CLASSES, COLS,
                         figsize=(COLS * 1.4, config.NUM_CLASSES * 1.4))
fig.suptitle(
    f"Colored MNIST Sample Grid  (correlation={config.CORRELATION})\n"
    "Each row = one digit class; rows share a dominant colour by design.",
    fontsize=10, y=1.01)

indices_by_label = {d: [] for d in range(config.NUM_CLASSES)}
for idx, lbl in enumerate(dataset.labels):
    indices_by_label[lbl].append(idx)

for row, digit in enumerate(range(config.NUM_CLASSES)):
    chosen = indices_by_label[digit][:COLS]
    for col, idx in enumerate(chosen):
        img, _ = dataset[idx]
        axes[row, col].imshow(img.permute(1, 2, 0).numpy())
        axes[row, col].axis("off")
        if col == 0:
            axes[row, col].set_ylabel(f"Digit {digit}", fontsize=8, labelpad=3)

plt.tight_layout()
p1 = os.path.join(config.FIGURES_DIR, "sample_grid.png")
plt.savefig(p1, dpi=120, bbox_inches="tight")
plt.close()
print(f"      Saved -> {p1}")

# ------------------------------------------------------------------ Figure 2: Colour heatmap
print("\n[3/4] Creating colour-distribution heatmap (Figure 2) ...")

COLOR_NAMES = ["Red","Green","Blue","Yellow","Purple",
               "Cyan","Orange","Lime","Pink","Violet"]

def detect_colour_index(img_np):
    """
    Cosine similarity between the image mean-RGB and each palette colour.
    Returns the palette index that is most similar.
    """
    mean_rgb = img_np.mean(axis=(0, 1)).astype(np.float32)
    best_idx, best_sim = 0, -1.0
    for key, colour in config.DIGIT_COLORS.items():
        pv   = np.array(colour, dtype=np.float32)
        dot  = float(np.dot(mean_rgb, pv))
        norm = np.linalg.norm(mean_rgb) * np.linalg.norm(pv) + 1e-8
        sim  = dot / norm
        if sim > best_sim:
            best_sim, best_idx = sim, key
    return best_idx

counts = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=np.float32)
for idx in range(len(dataset)):
    img_t, label = dataset[idx]
    img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    counts[int(label), detect_colour_index(img_np)] += 1

frac = counts / (counts.sum(axis=1, keepdims=True) + 1e-8)

fig2, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(frac, cmap="YlOrRd", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label="Fraction of samples")
ax.set_xticks(range(config.NUM_CLASSES))
ax.set_xticklabels([f"{c}\n({COLOR_NAMES[c]})" for c in range(10)], fontsize=8)
ax.set_yticks(range(config.NUM_CLASSES))
ax.set_yticklabels([f"Digit {d}" for d in range(10)], fontsize=9)
ax.set_xlabel("Detected colour applied to sample", fontsize=10)
ax.set_ylabel("True digit label", fontsize=10)
ax.set_title(
    f"Colour-Label Distribution  (correlation={config.CORRELATION})\n"
    "Diagonal near 0.90 = biased colour dominates each digit row.",
    fontsize=11)

for r in range(config.NUM_CLASSES):
    for c in range(config.NUM_CLASSES):
        v = frac[r, c]
        ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                fontsize=7, color="white" if v > 0.6 else "black")

plt.tight_layout()
p2 = os.path.join(config.FIGURES_DIR, "colour_distribution.png")
plt.savefig(p2, dpi=120, bbox_inches="tight")
plt.close()
print(f"      Saved -> {p2}")

# ------------------------------------------------------------------ Terminal summary
print("\n[4/4] Terminal correlation summary")
print("-" * 58)
print(f"  {'Digit':>5}  {'Colour':>10}  {'Fraction correct':>17}  Bar (out of 20)")
print("-" * 58)
for d in range(config.NUM_CLASSES):
    f = frac[d, d]
    bar = "#" * int(f * 20) + "." * (20 - int(f * 20))
    print(f"  {d:>5}  {COLOR_NAMES[d]:>10}  {f:>16.1%}  [{bar}]")
print("-" * 58)

avg = float(np.diag(frac).mean())
print(f"\n  Mean diagonal fraction : {avg:.1%}")
print(f"  (Expected around {config.CORRELATION:.0%} given CORRELATION={config.CORRELATION})")

print("\n" + "=" * 60)
print("VERIFICATION CHECKLIST")
print("=" * 60)
ok = "OK" if avg > 0.80 else "FAIL"
print(f"  [{ok}] Mean diagonal >= 80% -> {avg:.1%}")
print("  [ ] Open outputs/figures/sample_grid.png")
print("       Each digit row should share a dominant colour.")
print("  [ ] Open outputs/figures/colour_distribution.png")
print("       Diagonal cells dark (~0.90). Off-diagonal pale (~0.01).")
print()
print("Next step: python experiments/02_baseline_cnn.py")
