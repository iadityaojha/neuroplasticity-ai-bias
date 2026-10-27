# -*- coding: utf-8 -*-
"""
training/train.py
-----------------
Generic training loop used by all experiments.

Why keep the training loop separate from the model?
  Models and training procedures can change independently.
  Keeping them separate lets us reuse the same loop for the baseline
  AND the mitigated model in Step 3, with a single line change.
"""

import torch
import torch.nn as nn


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """
    Run one full pass over the training dataset.

    Returns
    -------
    dict with keys:
      loss     : average cross-entropy loss for this epoch
      accuracy : fraction of training samples correctly classified
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # ---- Forward pass ----
        logits = model(images)              # (batch, num_classes)
        loss   = criterion(logits, labels)

        # ---- Backward pass ----
        optimizer.zero_grad()   # clear gradients from previous batch
        loss.backward()         # compute gradients
        optimizer.step()        # update weights

        # ---- Track stats ----
        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss    += loss.item() * len(labels)
        total_samples += len(labels)

    return {
        "loss":     total_loss    / total_samples,
        "accuracy": total_correct / total_samples,
    }


def train(
    model: nn.Module,
    train_loader,
    epochs: int,
    lr: float,
    device: torch.device,
    verbose: bool = True,
) -> list:
    """
    Full training run over `epochs` epochs.

    Returns
    -------
    history : list of dicts, one per epoch, each containing loss and accuracy.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # ReduceLROnPlateau halves the learning rate if loss stops improving.
    # This is a safe default that works well for short experiments.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    history = []
    for epoch in range(1, epochs + 1):
        stats = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step(stats["loss"])
        history.append({"epoch": epoch, **stats})

        if verbose:
            print(f"  Epoch {epoch:>2}/{epochs}  "
                  f"loss={stats['loss']:.4f}  "
                  f"train_acc={stats['accuracy']:.1%}")

    return history
