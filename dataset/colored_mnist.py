# -*- coding: utf-8 -*-
"""
dataset/colored_mnist.py
------------------------
Generates a "Colored MNIST" dataset with a controlled spurious
colour-label correlation.

What is a spurious correlation?
  A spurious correlation is a statistical relationship that exists in
  training data but does NOT reflect a real causal rule in the world.
  Here: digit colour is correlated with digit identity in training data,
  but colour carries NO real information about what digit it is.

How we create it:
  1. Download grayscale MNIST.
  2. For each sample, look up the digit label (0-9).
  3. With probability `correlation`, apply that digit's "assigned" colour.
     With probability `1 - correlation`, apply a RANDOM other colour.
  4. The resulting RGB image is fed to the CNN.

Why 0.9 correlation?
  At 90%, colour is a very strong (but imperfect) predictor.
  A lazy model ignoring shape can still achieve ~90% accuracy by
  memorising colour -> digit mappings alone.
"""

import os, sys, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def set_all_seeds(seed: int) -> None:
    """
    Lock every random number generator used by the project.

    Why four different seeds?
      - Python random, NumPy, PyTorch CPU, and PyTorch CUDA each have
        their own RNG. Setting all four guarantees identical results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def colorize_digit(
    gray_image: np.ndarray,
    label: int,
    correlation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Turn a single grayscale digit image into an RGB coloured image.

    Parameters
    ----------
    gray_image  : shape (H, W), values in [0, 255], dtype uint8
    label       : the digit class (0-9)
    correlation : probability of applying the "biased" colour
    rng         : seeded NumPy generator

    Returns
    -------
    rgb_image : shape (H, W, 3), values in [0, 255], dtype uint8

    How it works:
      Bright pixels take on the chosen colour; dark background stays black.
    """
    if rng.random() < correlation:
        colour = config.DIGIT_COLORS[label]
    else:
        other_labels = [l for l in config.DIGIT_COLORS if l != label]
        colour = config.DIGIT_COLORS[int(rng.choice(other_labels))]

    fg = gray_image.astype(np.float32) / 255.0   # (H, W) in [0,1]
    r  = (fg * colour[0]).astype(np.uint8)
    g  = (fg * colour[1]).astype(np.uint8)
    b  = (fg * colour[2]).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)           # (H, W, 3)


class ColoredMNIST(Dataset):
    """
    PyTorch Dataset wrapping MNIST with controlled colour bias.

    Attributes
    ----------
    images : list of np.ndarray shape (H, W, 3)
    labels : list of int  -- digit labels (0-9)
    """

    def __init__(
        self,
        root="./data",
        train=True,
        num_samples=None,
        correlation=None,
        seed=None,
        transform=None,
    ):
        self.correlation = correlation if correlation is not None else config.CORRELATION
        self.seed        = seed        if seed        is not None else config.SEED
        self.transform   = transform

        raw = torchvision.datasets.MNIST(
            root=root, train=train, download=True,
            transform=T.ToTensor(),
        )

        n   = min(num_samples or len(raw), len(raw))
        rng = np.random.default_rng(self.seed)

        self.images = []
        self.labels = []

        for i in range(n):
            tensor, label = raw[i]
            # tensor: (1, H, W) in [0,1] -- squeeze to (H, W) in [0,255]
            gray = (tensor.squeeze(0).numpy() * 255).astype(np.uint8)
            self.images.append(colorize_digit(gray, label, self.correlation, rng))
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns (image_tensor, label).

        image_tensor shape: (3, H, W) in [0, 1] -- standard PyTorch CNN format.
        """
        img   = self.images[idx].astype(np.float32) / 255.0
        img   = torch.from_numpy(img).permute(2, 0, 1)  # (H,W,3) -> (3,H,W)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders(batch_size=None, seed=None, correlation=None):
    """
    Convenience function: returns (train_loader, test_loader).

    Both loaders use the SAME correlation so the baseline sees consistent
    evaluation conditions. In later steps we add an unbiased test set
    (correlation=0.0) to directly measure shape vs. colour reliance.
    """
    batch_size  = batch_size  or config.BATCH_SIZE
    seed        = seed        or config.SEED
    correlation = correlation if correlation is not None else config.CORRELATION

    train_ds = ColoredMNIST(train=True,  num_samples=config.DATASET_SIZE,
                            correlation=correlation, seed=seed)
    test_ds  = ColoredMNIST(train=False, num_samples=config.TEST_SIZE,
                            correlation=correlation, seed=seed + 1)

    def worker_init(wid):
        np.random.seed(seed + wid)
        random.seed(seed + wid)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, worker_init_fn=worker_init)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0)
    return train_loader, test_loader
