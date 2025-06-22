import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., use_topological_attention=False, use_gated_attention=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.use_topological_attention = use_topological_attention
        self.use_gated_attention = use_gated_attention

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.use_topological_attention or self.use_gated_attention:
            # Learnable parameters for the geometric bias
            self.alpha = nn.Parameter(torch.tensor(1.0))
            self.sigma = nn.Parameter(torch.tensor(1.0))

        if self.use_gated_attention:
            self.gate_controller = nn.Linear(dim_head, 1)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, klein_coords=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if self.use_topological_attention or self.use_gated_attention:
            if klein_coords is None:
                raise ValueError(
                    "klein_coords must be provided for topological/gated attention"
                )

            # Calculate Klein bottle distance and similarity bias
            coords1 = klein_coords.unsqueeze(2)
            coords2 = klein_coords.unsqueeze(1)
            d = torch.abs(coords1 - coords2)
            d = torch.min(d, 2 * np.pi - d)
            d_torus_sq = (d ** 2).sum(dim=-1)
            coords2_twisted = coords2.clone()
            coords2_twisted[..., 0] = (
                coords2_twisted[..., 0] + np.pi) % (2 * np.pi)
            coords2_twisted[..., 1] = (
                2 * np.pi - coords2_twisted[..., 1]) % (2 * np.pi)
            d_twisted = torch.abs(coords1 - coords2_twisted)
            d_twisted = torch.min(d_twisted, 2 * np.pi - d_twisted)
            d_twisted_torus_sq = (d_twisted ** 2).sum(dim=-1)
            dist_sq = torch.min(d_torus_sq, d_twisted_torus_sq)
            geom_bias = torch.exp(-dist_sq / (self.sigma ** 2))
            geom_bias = geom_bias.unsqueeze(1).repeat(1, self.heads, 1, 1)
            geom_bias_padded = F.pad(geom_bias, (1, 0, 1, 0), "constant", 0)

            if self.use_gated_attention:
                # Approach C: Gated mixing
                # Gate is calculated per query head
                # q has shape (b, h, n+1, d_head)
                # -> (b, h, n+1, 1)
                gate = torch.sigmoid(self.gate_controller(q))

                # Mix scores
                # Note: B_geom is just the similarity, not scores.
                # So we use it as a bias, but gated.
                gated_bias = gate * self.alpha * geom_bias_padded
                dots = dots + gated_bias

            elif self.use_topological_attention:
                # Approach B: Simple bias
                dots = dots + self.alpha * geom_bias_padded

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., use_topological_attention=False, use_gated_attention=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head,
                                       dropout=dropout, use_topological_attention=use_topological_attention,
                                       use_gated_attention=use_gated_attention)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, klein_coords=None):
        for attn, ff in self.layers:
            x = attn(x, klein_coords=klein_coords) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth,
                 heads, mlp_dim, pool='cls', channels=3, dim_head=64,
                 dropout=0., emb_dropout=0., use_klein_features=False,
                 use_topological_attention=False, use_gated_attention=False):
        super().__init__()
        if isinstance(image_size, tuple):
            image_height, image_width = image_size
        else:
            image_height, image_width = (image_size, image_size)

        if isinstance(patch_size, tuple):
            patch_height, patch_width = patch_size
        else:
            patch_height, patch_width = (patch_size, patch_size)

        assert image_height % patch_height == 0 and \
            image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * \
            (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, \
            'pool type must be either cls (class token) or mean (mean pooling)'

        self.use_klein_features = use_klein_features
        # Topological attention requires klein features to be passed in.
        self.use_topological_attention = use_topological_attention
        self.use_gated_attention = use_gated_attention
        if self.use_topological_attention or self.use_gated_attention:
            self.use_klein_features = True  # Force klein features on

        embedding_input_dim = patch_dim + 2 if self.use_klein_features and not (
            self.use_topological_attention or self.use_gated_attention) else patch_dim

        self.to_patch = Rearrange(
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=patch_height, p2=patch_width
        )
        self.patch_to_embedding = nn.Linear(embedding_input_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim,
                                       dropout,
                                       use_topological_attention=self.use_topological_attention,
                                       use_gated_attention=self.use_gated_attention)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, klein_features=None):
        x = self.to_patch(img)

        # This logic handles Approach A (concatenation)
        if self.use_klein_features and not (self.use_topological_attention or self.use_gated_attention):
            if klein_features is None:
                raise ValueError(
                    "klein_features must be provided when use_klein_features is True"
                )
            # klein_features are (b, num_patches, 2)
            # x is (b, num_patches, patch_dim)
            x = torch.cat((x, klein_features), dim=-1)

        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # This logic handles Approach B and C
        transformer_coords = klein_features if (
            self.use_topological_attention or self.use_gated_attention) else None
        x = self.transformer(x, klein_coords=transformer_coords)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
