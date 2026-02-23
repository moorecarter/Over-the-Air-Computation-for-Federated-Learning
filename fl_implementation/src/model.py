"""
MedViTV2-inspired model for BloodMNIST classification.

Simplified implementation focusing on the key architectural ideas:
- Vision Transformer backbone
- Adaptable for federated learning
- Support for both full MedViTV2-style and lightweight variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import timm


class KANLinear(nn.Module):
    """
    Simplified Kolmogorov-Arnold Network layer.
    Uses learnable B-spline basis functions for non-linear transformation.
    """

    def __init__(self, in_features: int, out_features: int, num_splines: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_splines = num_splines

        # Learnable spline coefficients
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, num_splines) * 0.1
        )
        # Base linear transformation
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * (2 / in_features) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Spline grid points
        self.register_buffer(
            'grid', torch.linspace(-1, 1, num_splines)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both 2D (batch, features) and 3D (batch, seq, features) input
        orig_shape = x.shape
        if x.dim() == 3:
            batch, seq, feat = x.shape
            x = x.reshape(batch * seq, feat)

        # Base linear transformation
        base_output = F.linear(x, self.base_weight, self.bias)

        # Spline-based non-linear transformation
        # Compute B-spline basis (simplified as Gaussian basis)
        x_expanded = x.unsqueeze(-1)  # (batch, in_features, 1)
        basis = torch.exp(-((x_expanded - self.grid) ** 2) / 0.5)  # (batch, in_features, num_splines)

        # Apply spline weights
        spline_output = torch.einsum('bik,oik->bo', basis, self.spline_weight)

        output = base_output + spline_output

        # Restore original shape if needed
        if len(orig_shape) == 3:
            output = output.reshape(orig_shape[0], orig_shape[1], -1)

        return output


class MLP(nn.Module):
    """Standard MLP block."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class KANMLP(nn.Module):
    """MLP block with KAN layers for enhanced non-linearity."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.kan1 = KANLinear(dim, hidden_dim)
        self.act = nn.GELU()
        self.kan2 = KANLinear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.kan1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.kan2(x)
        x = self.dropout(x)
        return x


class DilatedAttention(nn.Module):
    """
    Dilated Neighborhood Attention mechanism.
    Attends to dilated local neighborhoods for multi-scale feature capture.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        dilation: int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dilation = dilation

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Standard attention (can be extended to true dilated neighborhood attention)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TransformerBlock(nn.Module):
    """Transformer block with optional KAN-enhanced MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_kan: bool = True,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        dilation: int = 1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DilatedAttention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=dropout,
            dilation=dilation,
        )
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_kan:
            self.mlp = KANMLP(dim, mlp_hidden_dim, dropout=dropout)
        else:
            self.mlp = MLP(dim, mlp_hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MedViTV2(nn.Module):
    """
    MedViTV2: Vision Transformer with KAN and Dilated Attention for Medical Imaging.

    Args:
        img_size: Input image size
        patch_size: Patch size for tokenization
        in_chans: Number of input channels
        num_classes: Number of classification classes
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        use_kan: Whether to use KAN layers in MLP
        dropout: Dropout rate
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 8,  # BloodMNIST has 8 classes
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        use_kan: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer blocks with varying dilation
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_kan=use_kan,
                dropout=dropout,
                dilation=1 + (i % 3),  # Varying dilation: 1, 2, 3, 1, 2, 3...
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Classification head (use class token)
        x = self.head(x[:, 0])

        return x


def create_model(
    model_name: str = "medvit_small",
    num_classes: int = 8,
    img_size: int = 224,
    pretrained: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Create a model for BloodMNIST classification.

    Args:
        model_name: Model variant to create
            - "medvit_tiny": Lightweight for fast iteration
            - "medvit_small": Balanced performance/speed
            - "medvit_base": Full model
            - "timm_*": Use any TIMM model (e.g., "timm_vit_tiny_patch16_224")
        num_classes: Number of output classes
        img_size: Input image size
        pretrained: Whether to use pretrained weights (TIMM models only)

    Returns:
        PyTorch model
    """

    model_configs = {
        "medvit_tiny": dict(
            embed_dim=192, depth=4, num_heads=3, use_kan=True
        ),
        "medvit_small": dict(
            embed_dim=384, depth=6, num_heads=6, use_kan=True
        ),
        "medvit_base": dict(
            embed_dim=768, depth=12, num_heads=12, use_kan=True
        ),
        "medvit_small_no_kan": dict(
            embed_dim=384, depth=6, num_heads=6, use_kan=False
        ),
    }

    if model_name.startswith("timm_"):
        # Use TIMM model
        timm_name = model_name[5:]  # Remove "timm_" prefix
        model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size,
        )
    elif model_name in model_configs:
        config = model_configs[model_name]
        config.update(kwargs)
        model = MedViTV2(
            img_size=img_size,
            num_classes=num_classes,
            **config,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_configs.keys())}")

    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    for name in ["medvit_tiny", "medvit_small", "medvit_base"]:
        model = create_model(name, num_classes=8)
        params = count_parameters(model)

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        y = model(x)

        print(f"{name}: {params:,} params, output shape: {y.shape}")
