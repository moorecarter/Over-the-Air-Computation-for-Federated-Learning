"""
Generate a 4x2 grid showing one sample from each of the 8 BloodMNIST classes.
Saves the result as bloodmnist_classes.png.

Usage:
    python generate_bloodmnist_grid.py
"""

import numpy as np
import matplotlib.pyplot as plt
from medmnist import BloodMNIST

CLASS_NAMES = [
    "Basophil",
    "Eosinophil",
    "Erythrocyte",
    "Immature granulocyte",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelet",
]

def main():
    dataset = BloodMNIST(split="train", download=True, size=28)

    images = dataset.imgs       # (N, 28, 28, 3) uint8
    labels = dataset.labels.flatten()

    # Pick one sample per class
    samples = {}
    for idx, label in enumerate(labels):
        if label not in samples:
            samples[label] = images[idx]
        if len(samples) == 8:
            break

    fig, axes = plt.subplots(2, 4, figsize=(12, 7))
    fig.suptitle("BloodMNIST — 8 Blood Cell Classes", fontsize=20, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i])
        ax.set_title(CLASS_NAMES[i], fontsize=16, fontweight="bold")
        ax.axis("off")

    plt.subplots_adjust(top=0.88, hspace=0.35, wspace=0.15)
    plt.savefig("bloodmnist_classes.png", dpi=150)
    print("Saved bloodmnist_classes.png")
    plt.show()

if __name__ == "__main__":
    main()
