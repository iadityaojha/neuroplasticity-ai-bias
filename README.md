# Neuroplasticity in AI Bias

A small, reproducible PyTorch research experiment studying how neural networks
learn and bake in spurious correlations -- and how mitigation can help.

## Core Idea

We create a **Colored MNIST** dataset where digit colour is a *shortcut* to the
label (e.g. digit 0 is almost always red, digit 1 almost always green, etc.).
A naive CNN learns to rely on colour instead of digit shape.
We then measure and attempt to fix this bias.

## Project Structure

`
neuroplasticity-ai-bias/
|
+-- README.md                   <- You are here
+-- requirements.txt            <- Python dependencies
+-- config.py                   <- All hyperparameters and random seeds
|
+-- dataset/
|   +-- __init__.py
|   +-- colored_mnist.py        <- Dataset with spurious colour-label correlations
|
+-- models/
|   +-- __init__.py
|   +-- cnn.py                  <- Baseline CNN (Step 2)
|
+-- training/
|   +-- __init__.py
|   +-- train.py                <- Training loop (Step 2)
|   +-- evaluate.py             <- Evaluation metrics (Step 2)
|
+-- experiments/
|   +-- 01_dataset_check.py     <- STEP 1 (current): Generate and verify dataset
|   +-- 02_baseline_cnn.py      <- STEP 2: Train the biased CNN
|   +-- 03_mitigation.py        <- STEP 3: Apply a mitigation method
|   +-- 04_comparison.py        <- STEP 4: Before/after comparison
|
+-- outputs/
    +-- figures/                <- Saved plots
    +-- checkpoints/            <- Saved model weights
`

## Experiment Steps

| Step | Script                          | Status      |
|------|---------------------------------|-------------|
| 1    | experiments/01_dataset_check.py | Implemented |
| 2    | experiments/02_baseline_cnn.py  | Next        |
| 3    | experiments/03_mitigation.py    | Later       |
| 4    | experiments/04_comparison.py    | Later       |

## Reproducibility

All random seeds are set in config.py. The seed is printed at the start of
every script so results can always be traced back to exact settings.

## Quick Start

`ash
pip install -r requirements.txt
python experiments/01_dataset_check.py
`
