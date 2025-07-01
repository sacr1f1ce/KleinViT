import torch
import numpy as np


class KleinFeatureExtractor:
    """Feature extractor that maps image patches to Klein bottle coordinates
    using a bank of analytical Klein filters.
    """

    def __init__(self, patch_size: int = 3, num_angles: int = 4,
                 device: str = "cpu") -> None:
        """Create a bank of Klein filters.

        Args:
            patch_size: Size of each square patch / filter.
            num_angles: Number of discrete angles per axis. Total
                filters = num_angles ** 2 (matches KF_Layer where
                ``angle = int(sqrt(slices))``).
            device: Target torch device string.
        """
        self.patch_size = patch_size
        self.num_angles = num_angles
        self.device = torch.device(device)

        # Angular grids matching implementation in `topological_conv_layer`.
        # Values are evenly spaced but *exclude* the end-point (π or 2π),
        # because the original code uses `range(angle)` rather than
        # `linspace`. This ensures we generate exactly the same orientations
        # as in `KF_Layer` / `generate_klein_filter`.
        thetas1 = (torch.arange(num_angles, device=self.device)
                   * (np.pi / num_angles))
        thetas2 = (torch.arange(num_angles, device=self.device)
                   * (2.0 * np.pi / num_angles))

        kernels, theta_pairs = [], []
        for theta2 in thetas2:
            for theta1 in thetas1:
                k = self._generate_klein_filter(theta1, theta2)
                # Zero-mean / unit-variance for projection similarity.
                k = (k - k.mean()) / (k.std() + 1e-6)
                kernels.append(k)
                theta_pairs.append(torch.tensor([theta1, theta2],
                                                device=self.device))

        self.klein_kernels = torch.stack(kernels).unsqueeze(1)  # (F,1,p,p)
        self.thetas = torch.stack(theta_pairs)                  # (F,2)
        self.num_filters = self.klein_kernels.shape[0]

        # Flattened view for fast projection
        self._kernels_flat = self.klein_kernels.view(
            self.num_filters, -1).t()  # (p*p, F)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def get_klein_coordinates(self, image_batch: torch.Tensor) -> torch.Tensor:
        """Return (theta1, theta2) for every non-overlapping patch.

        Args:
            image_batch: (B, C, H, W) tensor in [0,1].
        Returns:
            Tensor of shape (B, N, 2) where N = num patches per image.
        """
        patches_ac_flat = self._extract_patches_ac_flat(image_batch)
        similarities = torch.matmul(
            patches_ac_flat, self._kernels_flat)  # (B,N,F)
        best_idx = torch.argmax(torch.abs(similarities),
                                dim=-1)          # (B,N)
        # (B,N,2)
        coords = self.thetas[best_idx]
        return coords

    def replace_patches_with_klein(self, image: torch.Tensor) -> torch.Tensor:
        """Return grayscale image where each patch is replaced by the
        closest Klein filter from the bank.
        """
        patches_ac_flat = self._extract_patches_ac_flat(
            image.unsqueeze(0))  # (1,N,p*p)
        similarities = torch.matmul(
            patches_ac_flat, self._kernels_flat)     # (1,N,F)
        best_idx = torch.argmax(torch.abs(similarities),
                                dim=-1).squeeze(0)  # (N,)
        selected = self.klein_kernels[best_idx].squeeze(1)  # (N,p,p)

        # Scale filters to [0,1] for visualization
        min_val = selected.amin(dim=(-2, -1), keepdim=True)
        max_val = selected.amax(dim=(-2, -1), keepdim=True)
        selected_norm = (selected - min_val) / (max_val - min_val + 1e-6)

        # Reassemble image
        c, h, w = image.shape
        p = self.patch_size
        num_h, num_w = h // p, w // p
        assembled = selected_norm.view(num_h, num_w, p, p)
        assembled = assembled.permute(0, 2, 1, 3).contiguous().view(h, w)
        return assembled.cpu()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_patches_ac_flat(self, images: torch.Tensor) -> torch.Tensor:
        """Return zero-mean flattened grayscale patches."""
        device = self.device
        images = images.to(device)
        b, c, h, w = images.shape
        p = self.patch_size
        patches = images.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B,N,C,p,p)
        patches_gray = patches.mean(dim=3, keepdim=True)  # (B,N,1,p,p)

        # Zero-mean per patch
        patch_means = patches_gray.mean(dim=(-1, -2, -3), keepdim=True)
        patches_ac = patches_gray - patch_means

        # Unit-std (avoid division by zero)
        patch_std = patches_ac.flatten(-3).std(dim=-1, keepdim=True)
        patch_std = (patch_std
                     .unsqueeze(-1)
                     .unsqueeze(-1))  # (B,N,1,1,1)
        patches_norm = patches_ac / (patch_std + 1e-6)

        return patches_norm.view(b, -1, p * p)

    def _generate_klein_filter(self, theta1: torch.Tensor,
                               theta2: torch.Tensor) -> torch.Tensor:
        """Generate one Klein filter on a regular grid (approx.)."""
        p = self.patch_size
        y = torch.linspace(-1.0, 1.0, steps=p, device=self.device)
        x = torch.linspace(-1.0, 1.0, steps=p, device=self.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        def _Q(t):
            return (2 * t) ** 2 - 1

        klein_val = (
            torch.sin(theta2)
            * (torch.cos(theta1) * xx + torch.sin(theta1) * yy)
            + torch.cos(theta2)
            * _Q(torch.cos(theta1) * xx + torch.sin(theta1) * yy)
        )
        return klein_val.float()
