from src.utils import visualize_gabor_kernels, visualize_reconstruction
from src.klein_tools.feature_extractor import KleinFeatureExtractor
import torch
import torchvision
import os
import random

# Ensure the output directory exists
os.makedirs("output", exist_ok=True)


def generate():
    """
    Generates Gabor kernel visualizations and 3-panel reconstructions for
    a few random sample CIFAR-10 images.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 1. Initialize Extractor and Visualize Kernels ---
    print("==> Initializing feature extractor...")
    patch_size = 3
    extractor = KleinFeatureExtractor(
        patch_size=patch_size,
        num_orientations=16,
        device=device
    )
    visualize_gabor_kernels(extractor)

    # --- 2. Prepare Dataset ---
    print("\n==> Preparing CIFAR-10 data...")
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=torchvision.transforms.ToTensor()
    )
    class_names = dataset.classes

    # --- 3. Generate Reconstructions for 3 Random Images ---
    num_samples_to_show = 3
    random_indices = random.sample(range(len(dataset)), num_samples_to_show)

    for index in random_indices:
        image, label_idx = dataset[index]
        name = class_names[label_idx].lower()

        print(
            f"\n-- Generating visualization for: {name.capitalize()} (Image #{index}) --")

        # Pad image to be divisible by patch size
        padded_image = torchvision.transforms.Pad((0, 0, 1, 1))(image)

        visualize_reconstruction(
            image_tensor=padded_image,
            extractor=extractor,
            class_name=name,
            save_path=f"output/reconstruction_{name}_{index}.png"
        )


if __name__ == "__main__":
    generate()
