from src.klein_tools.feature_extractor import KleinFeatureExtractor
import torchvision.datasets as dsets
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch
import os
from typing import Optional
from src.utils import visualize_klein_kernels
import torch.nn.functional as F


DATA_DIR = "data"          # Path where CIFAR-10 is stored
PATCH_SIZE = 3             # Patch / filter size (must divide 32)
NUM_ANGLES = 4            # Angular samples per axis
INDICES = [1, 10, 1000]    # CIFAR-10 indices to visualise

os.makedirs("output", exist_ok=True)


def _pad_to_multiple(t: torch.Tensor, multiple: int) -> torch.Tensor:
    """Pad HxW image tensor (C,H,W or 1,H,W) on right/bottom to be multiple."""
    _, h, w = t.shape
    pad_bottom = (multiple - h % multiple) % multiple
    pad_right = (multiple - w % multiple) % multiple
    if pad_bottom == 0 and pad_right == 0:
        return t
    pad = (0, pad_right, 0, pad_bottom)  # (left,right,top,bottom)
    return F.pad(t, pad, value=0.0)


def generate(sample_idx: int, device: Optional[str] = None) -> None:
    """Create a 3-panel visualization for a single CIFAR-10 image."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[IDX {sample_idx}] using device: {device}")

    # Load CIFAR-10 (test split)
    cifar = dsets.CIFAR10(root=DATA_DIR, train=False, download=False)
    if sample_idx >= len(cifar):
        print(f"Index {sample_idx} is out of range. Skipping.")
        return

    img_pil, label = cifar[sample_idx]
    class_name = cifar.classes[label]

    # Convert to tensors
    to_tensor = T.ToTensor()
    img_tensor = to_tensor(img_pil)             # (3, H, W)
    gray_tensor = T.Grayscale()(img_tensor)     # (1, H, W)

    # Pad to make dimensions divisible by PATCH_SIZE
    img_tensor = _pad_to_multiple(img_tensor, PATCH_SIZE)
    gray_tensor = _pad_to_multiple(gray_tensor, PATCH_SIZE)

    # Klein patch replacement
    extractor = KleinFeatureExtractor(patch_size=PATCH_SIZE,
                                      num_angles=NUM_ANGLES,
                                      device=device)
    klein_img = extractor.replace_patches_with_klein(img_tensor)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"{class_name} | patch={PATCH_SIZE}", fontsize=14)

    axes[0].imshow(img_tensor.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Color")

    axes[1].imshow(gray_tensor.squeeze().cpu().numpy(), cmap="gray")
    axes[1].set_title("Grayscale")

    axes[2].imshow(klein_img.numpy(), cmap="gray")
    axes[2].set_title("Klein patches")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    out_name = f"cifar_{class_name}_{sample_idx}_vis.png"
    out_path = os.path.join("output", out_name)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved visualization to {out_path}")
    plt.close()


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Save Klein filter-bank visualization once
    extractor_for_kernels = KleinFeatureExtractor(
        patch_size=PATCH_SIZE,
        num_angles=NUM_ANGLES,
        device=default_device,
    )
    kernels_path = f"output/klein_kernels_p{PATCH_SIZE}_a{NUM_ANGLES}.png"
    visualize_klein_kernels(extractor_for_kernels, save_path=kernels_path)

    # Generate per-image visualizations
    for idx in INDICES:
        generate(idx, device=default_device)
