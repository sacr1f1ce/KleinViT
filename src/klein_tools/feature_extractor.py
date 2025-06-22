import torch
import torch.nn.functional as F
import numpy as np


class KleinFeatureExtractor:
    """
    A class to handle the extraction of Klein bottle features from image patches.
    It creates a Gabor filter bank and uses it to project image patches to
    a feature space, representing each patch with Klein bottle coordinates
    (theta, phi).
    """

    def __init__(self, patch_size=3, num_orientations=16, device='cpu'):
        self.patch_size = patch_size
        self.num_orientations = num_orientations
        self.device = device
        self.orientations = torch.arange(
            num_orientations, device=device
        ) * (np.pi / num_orientations)

        gabor_kernels_complex = [
            self._create_gabor_kernel(th, 0.6, 0.5, patch_size, device)
            for th in self.orientations
        ]

        # We use the real part of the Gabor filters for our projection
        self.gabor_kernels = torch.stack(
            [torch.real(k) for k in gabor_kernels_complex], dim=0
        ).unsqueeze(1)  # Shape: (num_orientations, 1, patch_size, patch_size)

        # Ensure kernels are on the correct device
        self.gabor_kernels = self.gabor_kernels.to(device)

    def get_klein_coordinates(self, image_batch):
        """
        Extracts Klein bottle coordinates (theta, phi) for each patch in a
        batch of images.

        Args:
            image_batch (torch.Tensor): A batch of images of shape
                                        (b, c, h, w).

        Returns:
            torch.Tensor: A tensor of shape (b, num_patches, 2) containing
                          the (theta, phi) coordinates for each patch.
        """
        b, c, h, w = image_batch.shape
        p = self.patch_size

        # 1. Decompose image into patches
        patches = image_batch.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(b, -1, c, p, p)  # (b, num_patches, c, p, p)

        # Convert patches to grayscale for Gabor projection
        # (b, num_patches, 1, p, p)
        patches_gray = patches.mean(dim=2, keepdim=True)

        # 2. Decompose patches into AC components
        patch_means = patches_gray.mean(dim=(-3, -2, -1), keepdim=True)
        patches_ac = patches_gray - patch_means

        # Flatten for matmul
        # The new channel dim is 1
        patches_ac_flat = patches_ac.view(
            b, -1, 1 * p * p)  # (b, num_patches, p*p)
        kernels_flat = self.gabor_kernels.view(
            self.num_orientations, -1
        ).t()  # (p*p, num_orientations)

        # 3. Project AC components onto Gabor filters
        projection_magnitudes = torch.matmul(
            patches_ac_flat, kernels_flat
        )  # (b, num_patches, num_orientations)

        # 4. Find the best orientation (theta) for each patch
        best_kernel_indices = torch.argmax(
            projection_magnitudes, dim=-1
        )  # (b, num_patches)
        theta = self.orientations[best_kernel_indices]  # (b, num_patches)

        # 5. Determine the phase (phi) for each patch
        best_projection_magnitudes = torch.gather(
            projection_magnitudes, 2, best_kernel_indices.unsqueeze(-1)
        ).squeeze(-1)  # (b, num_patches)

        patch_ac_norms = torch.norm(
            patches_ac_flat, dim=-1)  # (b, num_patches)

        # Clamp for numerical stability
        cosine_of_phi = torch.clamp(
            best_projection_magnitudes / (patch_ac_norms + 1e-9), -1.0, 1.0
        )
        phi = torch.acos(cosine_of_phi)  # (b, num_patches)

        # 6. Stack coordinates
        klein_coords = torch.stack([theta, phi], dim=-1)

        return klein_coords

    def _create_gabor_kernel(self, theta, frequency, sigma, kernel_size, device):
        y, x = torch.meshgrid(
            torch.arange(-kernel_size // 2 + 1,
                         kernel_size // 2 + 1, device=device),
            torch.arange(-kernel_size // 2 + 1,
                         kernel_size // 2 + 1, device=device),
            indexing='ij'
        )
        x_theta = x * torch.cos(theta) + y * torch.sin(theta)
        y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

        gb = torch.exp(-.5 * (x_theta ** 2 / sigma ** 2 + y_theta ** 2 / sigma ** 2)) * \
            torch.exp(1j * 2 * np.pi * frequency * x_theta)
        return (gb - gb.mean()) / gb.std()


def reconstruct_patch_from_gabor(patch, extractor):
    """
    Reconstructs a single image patch using the Gabor filter bank from the
    feature extractor. This function is mostly for visualization.
    """
    # Decompose into DC (brightness) and AC (texture)
    dc_component = patch.mean()
    ac_component = patch - dc_component

    # Flatten for projection
    ac_component_flat = ac_component.flatten()

    # Project onto Gabor filters
    kernels_flat = extractor.gabor_kernels.view(
        extractor.num_orientations, -1
    )
    projection_magnitudes = torch.matmul(ac_component_flat, kernels_flat.t())

    # Find best matching filter
    best_kernel_idx = torch.argmax(projection_magnitudes)
    best_kernel = kernels_flat[best_kernel_idx]
    projection_on_best = projection_magnitudes[best_kernel_idx]

    # Reconstruct the AC component
    reconstructed_ac = best_kernel.view(patch.shape) * projection_on_best
    reconstructed_ac_normalized = reconstruct_ac_from_texture(
        patch, extractor
    )

    # Reconstruct the full patch
    reconstructed_full = reconstructed_ac_normalized + dc_component
    reconstructed_full = torch.clamp(reconstructed_full, 0, 1)

    return reconstructed_ac_normalized, reconstructed_full


def reconstruct_ac_from_texture(patch, extractor):
    """Helper to reconstruct just the texture component."""
    dc, ac = patch.mean(), patch - patch.mean()
    ac_flat = ac.flatten()
    kernels_flat = extractor.gabor_kernels.view(extractor.num_orientations, -1)
    projections = torch.matmul(ac_flat, kernels_flat.t())
    best_idx = torch.argmax(projections)
    reconstructed_ac = kernels_flat[best_idx].view(
        patch.shape) * projections[best_idx]

    # Normalize for visualization
    min_val, max_val = torch.min(reconstructed_ac), torch.max(reconstructed_ac)
    if max_val > min_val:
        final_patch_real = (reconstructed_ac - min_val) / (max_val - min_val)
        return final_patch_real

    return torch.zeros_like(reconstructed_ac)
