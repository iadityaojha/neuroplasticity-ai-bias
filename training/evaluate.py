# -*- coding: utf-8 -*-
"""
training/evaluate.py
---------------------
Evaluation metrics for all experiments.

We compute THREE things:
  1. Overall accuracy    -- single number, easy to compare
  2. Per-class accuracy  -- reveals which digits the model gets wrong
  3. Confusion matrix    -- shows which digits get confused with which

Why per-class accuracy matters for bias research:
  A model might have 90% overall accuracy but only 60% on digit 1
  because digit 1 shares a colour with some other class. Overall
  accuracy hides this; per-class accuracy exposes it.
"""

import torch
import torch.nn as nn
import numpy as np


def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int = 10,
) -> dict:
    """
    Evaluate a trained model on a data loader.

    Returns
    -------
    dict with keys:
      accuracy        : float  -- overall fraction correct
      per_class_acc   : np.ndarray shape (num_classes,) -- per-digit accuracy
      confusion_matrix: np.ndarray shape (num_classes, num_classes)
                        confusion_matrix[true][pred] = count
      loss            : float  -- average cross-entropy
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    # confusion_matrix[true_label][predicted_label] = count
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():   # no gradient computation needed during evaluation
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)
            preds  = logits.argmax(dim=1)

            total_correct += (preds == labels).sum().item()
            total_loss    += loss.item() * len(labels)
            total_samples += len(labels)

            # Fill confusion matrix
            for true, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                conf_mat[true][pred] += 1

    # Per-class accuracy: diagonal / row sum
    row_sums      = conf_mat.sum(axis=1)                         # (num_classes,)
    per_class_acc = np.where(
        row_sums > 0,
        conf_mat.diagonal() / row_sums,
        0.0
    )

    return {
        "accuracy":         total_correct / total_samples,
        "loss":             total_loss    / total_samples,
        "per_class_acc":    per_class_acc,
        "confusion_matrix": conf_mat,
    }
