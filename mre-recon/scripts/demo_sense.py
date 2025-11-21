#!/usr/bin/env python3
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import src/ modules
sys.path.append(os.path.abspath("."))

from src.sense.sense_operator import SenseOperator


def main():
    print("\n--- Running SENSE Demo ---\n")

    # ----------------------------
    # Synthetic coil sensitivity maps
    # ----------------------------
    ny, nx = 64, 64
    n_vox = ny * nx

    y, x = np.mgrid[-1:1:ny*1j, -1:1:nx*1j]

    # 4 simulated coils at corners
    coils = np.stack([
        np.exp(-((x-0.5)**2 + (y-0.5)**2) * 8),
        np.exp(-((x+0.5)**2 + (y-0.5)**2) * 8),
        np.exp(-((x-0.5)**2 + (y+0.5)**2) * 8),
        np.exp(-((x+0.5)**2 + (y+0.5)**2) * 8),
    ], axis=0).astype(np.complex64)

    ncoils = coils.shape[0]
    print(f"Generated {ncoils} synthetic coils.")

    # ----------------------------
    # Build SENSE operator
    # ----------------------------
    S = SenseOperator(coils)
    print(f"SENSE operator shape: {S.shape}")

    # ----------------------------
    # Example image
    # ----------------------------
    img = np.zeros((ny, nx), dtype=np.complex64)
    img[24:40, 24:40] = 1.0  # bright square for visualization

    # ----------------------------
    # Forward: generate coil images
    # ----------------------------
    y_data = S.forward(img)        # shape = (ncoils * ny * nx,)
    coil_imgs = y_data.reshape(ncoils, ny, nx)

    # ----------------------------
    # Adjoint: combine images
    # ----------------------------
    recon = S.adjoint(y_data).reshape(ny, nx)

    print("Forward and adjoint completed.")
    print("coil_imgs shape:", coil_imgs.shape)
    print("recon shape:", recon.shape)

    # ----------------------------
    # Plot results
    # ----------------------------

    # Coil sensitivity maps
    fig, ax = plt.subplots(1, ncoils, figsize=(15, 4))
    for i in range(ncoils):
        ax[i].imshow(np.abs(coils[i]), cmap="viridis")
        ax[i].set_title(f"Coil Sensitivity {i+1}")
        ax[i].axis("off")
    plt.suptitle("SENSE Coil Sensitivity Maps")
    plt.tight_layout()
    plt.show()

    # Forward coil images
    fig, ax = plt.subplots(1, ncoils, figsize=(15, 4))
    for i in range(ncoils):
        ax[i].imshow(np.abs(coil_imgs[i]), cmap="magma")
        ax[i].set_title(f"Coil Image {i+1}")
        ax[i].axis("off")
    plt.suptitle("Forward Coil Images")
    plt.tight_layout()
    plt.show()

    # Reconstructed adjoint image
    plt.figure(figsize=(5,5))
    plt.imshow(np.abs(recon), cmap="gray")
    plt.title("Adjoint Reconstruction")
    plt.axis("off")
    plt.show()

    print("\nSENSE demo complete.\n")


if __name__ == "__main__":
    main()
