import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

from src.klein_tools.feature_extractor import KleinFeatureExtractor


def visualize_gabor_kernels(extractor, save_path="output/gabor_kernels.png"):
    """
    Visualizes the real parts of the Gabor filters in the extractor.
    """
    n_orientations = extractor.num_orientations
    cols = int(np.ceil(np.sqrt(n_orientations)))
    rows = int(np.ceil(n_orientations / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, ax in enumerate(axes.flat):
        if i < len(extractor.gabor_kernels_complex):
            # Visualize the real part of the complex kernel
            kernel = torch.real(
                extractor.gabor_kernels_complex[i]).cpu().numpy()
            ax.imshow(kernel, cmap='gray')
            ax.set_title(f"{extractor.orientations[i].item():.2f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Gabor Filter Bank (Real Parts)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved Gabor filter bank plot to {save_path}")
    plt.close()


def visualize_reconstruction(image_tensor, extractor, class_name,
                             save_path="output/reconstruction.png"):
    """
    Generates a 3-panel visualization showing the original color image,
    a reconstruction from texture (AC), and texture + brightness (AC + DC).
    """
    device = extractor.device
    patch_size = extractor.patch_size
    image_tensor = image_tensor.to(device)
    image_height, image_width = image_tensor.shape[1:]

    image_tensor_gray = torchvision.transforms.Grayscale()(image_tensor)

    color_patches = image_tensor.unfold(1, patch_size, patch_size) \
        .unfold(2, patch_size, patch_size) \
        .contiguous().view(-1, 3, patch_size, patch_size)

    gray_patches = image_tensor_gray.unfold(1, patch_size, patch_size) \
        .unfold(2, patch_size, patch_size) \
        .contiguous().view(-1, 1, patch_size, patch_size)

    all_coords = extractor.extract_coords(color_patches)

    patches_ac, patches_ac_dc = [], []
    max_magnitude = max(c['magnitude']
                        for c in all_coords) if all_coords else 1

    for i, coords in enumerate(all_coords):
        ideal_patch = extractor.generate_patch(
            coords['theta'], coords['phi'], coords['polarity'])

        texture_intensity = (coords['magnitude'] / max_magnitude) * 0.5
        ac_patch = (ideal_patch - 0.5) * texture_intensity
        patches_ac.append(ac_patch)

        dc_component = gray_patches[i].mean()
        ac_dc_patch = torch.clamp(ac_patch + dc_component, 0, 1)
        patches_ac_dc.append(ac_dc_patch)

    num_h, num_w = image_height // patch_size, image_width // patch_size
    recon_ac_tensor = torch.stack(patches_ac).reshape(
        num_h, num_w, patch_size, patch_size)
    recon_image_ac = rearrange(recon_ac_tensor, 'h w p1 p2 -> (h p1) (w p2)')
    recon_ac_dc_tensor = torch.stack(patches_ac_dc).reshape(
        num_h, num_w, patch_size, patch_size)
    recon_image_ac_dc = rearrange(
        recon_ac_dc_tensor, 'h w p1 p2 -> (h p1) (w p2)')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Reconstruction for: {class_name.capitalize()}', fontsize=20)

    axes[0].imshow(image_tensor.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original Color", fontsize=16)

    axes[1].imshow(recon_image_ac.squeeze().cpu().numpy(), cmap='gray')
    axes[1].set_title("Reconstruction (Texture Only)", fontsize=16)

    axes[2].imshow(recon_image_ac_dc.squeeze().cpu().numpy(), cmap='gray')
    axes[2].set_title("Reconstruction (Texture + Brightness)", fontsize=16)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved reconstruction for '{class_name}' to {save_path}")
    plt.close()
