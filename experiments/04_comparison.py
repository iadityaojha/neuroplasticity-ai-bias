# -*- coding: utf-8 -*-
"""
experiments/04_comparison.py
-----------------------------
STEP 4 -- Side-by-side before/after comparison.

Loads both checkpoints, evaluates on identical test sets,
produces the final summary figure.

How to run:
  python experiments/04_comparison.py
"""

import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import config
from dataset.colored_mnist import ColoredMNIST, set_all_seeds
from models.cnn import BaselineCNN
from training.evaluate import evaluate

set_all_seeds(config.SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLOR_NAMES = ['Red','Green','Blue','Yellow','Purple',
               'Cyan','Orange','Lime','Pink','Violet']

print("=" * 65)
print("STEP 4: Before / After Comparison")
print("=" * 65)

os.makedirs(config.FIGURES_DIR, exist_ok=True)

# ---- Load models ---------------------------------------------------------
def load_model(ckpt_name):
    m = BaselineCNN(num_classes=config.NUM_CLASSES).to(device)
    path = os.path.join(config.CHECKPOINTS_DIR, ckpt_name)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

print("\n[1/4] Loading checkpoints ...")
baseline  = load_model("baseline_cnn.pt")
mitigated = load_model("mitigated_cnn.pt")
print("      Both checkpoints loaded.")

# ---- Test sets -----------------------------------------------------------
print("\n[2/4] Building test sets ...")
biased_ds = ColoredMNIST(
    train=False, num_samples=config.TEST_SIZE,
    correlation=config.CORRELATION, seed=config.SEED + 1,
)
unbiased_ds = ColoredMNIST(
    train=False, num_samples=config.TEST_SIZE,
    correlation=0.0, seed=config.SEED + 99,
)
biased_loader   = DataLoader(biased_ds,   batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=0)
unbiased_loader = DataLoader(unbiased_ds, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=0)

# ---- Evaluate both -------------------------------------------------------
print("\n[3/4] Evaluating both models ...")

b_base = evaluate(baseline,  biased_loader,   device, config.NUM_CLASSES)
u_base = evaluate(baseline,  unbiased_loader, device, config.NUM_CLASSES)
b_mit  = evaluate(mitigated, biased_loader,   device, config.NUM_CLASSES)
u_mit  = evaluate(mitigated, unbiased_loader, device, config.NUM_CLASSES)

bias_baseline  = b_base['accuracy'] - u_base['accuracy']
bias_mitigated = b_mit['accuracy']  - u_mit['accuracy']
bias_reduction = bias_baseline - bias_mitigated

print()
print("  " + "=" * 60)
print(f"  {'Metric':<32} {'Baseline':>12} {'Mitigated':>12}")
print("  " + "-" * 60)
print(f"  {'Biased test accuracy':<32} {b_base['accuracy']:>11.1%} {b_mit['accuracy']:>11.1%}")
print(f"  {'Unbiased test accuracy':<32} {u_base['accuracy']:>11.1%} {u_mit['accuracy']:>11.1%}")
print(f"  {'Bias score (gap)':<32} {bias_baseline:>+11.1%} {bias_mitigated:>+11.1%}")
print("  " + "-" * 60)
print(f"  {'Bias reduction':<32} {'':>12} {bias_reduction:>+11.1%}")
print("  " + "=" * 60)
print()

if bias_reduction > 0.05:
    verdict = f"Grayscale augmentation REDUCED bias by {bias_reduction:.1%}."
elif bias_reduction > 0:
    verdict = f"Grayscale augmentation gave a small improvement ({bias_reduction:.1%})."
else:
    verdict = "Grayscale augmentation did NOT reduce bias in this run."
print(f"  Verdict: {verdict}")
print()

# ---- Figures -------------------------------------------------------------
print("[4/4] Saving comparison figures ...")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Panel 1: Overall accuracy
labels = ['Biased test', 'Unbiased test']
x = np.arange(len(labels))
w = 0.35
axes[0].bar(x - w/2, [b_base['accuracy']*100, u_base['accuracy']*100],
            w, label='Baseline',  color='#e05c5c', alpha=0.85)
axes[0].bar(x + w/2, [b_mit['accuracy']*100,  u_mit['accuracy']*100],
            w, label='Mitigated', color='#5cb85c', alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
axes[0].set_ylabel('Accuracy (%)'); axes[0].set_ylim(70, 102)
axes[0].set_title('Overall Accuracy', fontweight='bold')
axes[0].legend(fontsize=9); axes[0].grid(axis='y', alpha=0.3)
for bar in axes[0].patches:
    h = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.3,
                 f'{h:.1f}%', ha='center', va='bottom', fontsize=8)

# Panel 2: Bias scores
bias_vals = [bias_baseline * 100, bias_mitigated * 100]
bars = axes[1].bar(['Baseline','Mitigated'], bias_vals,
                   color=['#e05c5c','#5cb85c'], alpha=0.85, width=0.4)
axes[1].set_ylabel('Bias score (%)')
axes[1].set_title('Bias Score (lower is better)', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_ylim(0, max(bias_vals) * 1.5 + 1)
for bar, v in zip(bars, bias_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.2,
                 f'{v:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Panel 3: Per-class unbiased accuracy
x3 = np.arange(config.NUM_CLASSES)
axes[2].bar(x3 - w/2, u_base['per_class_acc'] * 100,
            w, label='Baseline',  color='#e05c5c', alpha=0.8)
axes[2].bar(x3 + w/2, u_mit['per_class_acc']  * 100,
            w, label='Mitigated', color='#5cb85c', alpha=0.8)
axes[2].set_xticks(x3)
axes[2].set_xticklabels([f'{d}' for d in range(10)], fontsize=9)
axes[2].set_ylabel('Accuracy on unbiased test (%)')
axes[2].set_title('Per-class Accuracy (Unbiased Test)', fontweight='bold')
axes[2].legend(fontsize=9); axes[2].grid(axis='y', alpha=0.3)
axes[2].set_ylim(50, 105)

fig.suptitle(
    f'Neuroplasticity in AI Bias -- Before vs After Mitigation\n'
    f'Baseline bias: {bias_baseline:+.1%}   |   '
    f'Mitigated bias: {bias_mitigated:+.1%}   |   '
    f'Reduction: {bias_reduction:+.1%}',
    fontsize=11, fontweight='bold', y=1.01,
)
plt.tight_layout()
p = os.path.join(config.FIGURES_DIR, 'final_comparison.png')
plt.savefig(p, dpi=130, bbox_inches='tight'); plt.close()
print(f"      Saved -> {p}")

# ---- Final summary -------------------------------------------------------
print()
print("=" * 65)
print("EXPERIMENT COMPLETE -- FINAL SUMMARY")
print("=" * 65)
print()
print(f"  Dataset       : Colored MNIST, {config.DATASET_SIZE} train / {config.TEST_SIZE} test")
print(f"  Correlation   : {config.CORRELATION}  (spurious colour-label bias)")
print(f"  Seed          : {config.SEED}")
print()
print("  BASELINE CNN")
print(f"    Biased test   : {b_base['accuracy']:.1%}")
print(f"    Unbiased test : {u_base['accuracy']:.1%}")
print(f"    Bias score    : {bias_baseline:+.1%}")
print()
print("  MITIGATED CNN (50% grayscale augmentation)")
print(f"    Biased test   : {b_mit['accuracy']:.1%}")
print(f"    Unbiased test : {u_mit['accuracy']:.1%}")
print(f"    Bias score    : {bias_mitigated:+.1%}")
print()
print(f"  BIAS REDUCTION : {bias_reduction:+.1%}")
print(f"  VERDICT        : {verdict}")
print()
print("  All outputs in outputs/figures/ and outputs/checkpoints/")
