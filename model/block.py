import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvolutionBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ConvolutionConcatBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvolutionConcatBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.GELU(),
        )

    def forward(self, x1, x2):
        x = torch.cat(tensors=[x1, x2], dim=1)
        out = self.block(x)
        return out


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.patcher = nn.Conv2d(
            in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0, 2, 1)


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int = 768, num_heads: int = 12, attn_dropout: int = 0):
        super().__init__()
        self.Block_norm = nn.BlockNorm(normalized_shape=embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )

    def forward(self, x):
        x = self.Block_norm(x)
        attn_output, _ = self.multihead_attn(query=x, key=x, value=x, need_weights=False)
        return attn_output


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int = 768, mlp_size: int = 3072, dropout: int = 0.1):
        super().__init__()
        self.Block_norm = nn.BlockNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        x = self.Block_norm(x)
        x = self.mlp(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 12,
        mlp_size: int = 3072,
        mlp_dropout: int = 0.1,
        attn_dropout: int = 0,
    ):
        super().__init__()
        self.msa_block = MultiHeadSelfAttentionBlock(
            embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout
        )
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x


class DeepCurveEstimationTransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 12,
        mlp_size: int = 3072,
        mlp_dropout: int = 0.1,
        attn_dropout: int = 0,
    ):
        super().__init__()
        self.transformer_encoder = TransformerEncoderBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_size=mlp_size,
            mlp_dropout=mlp_dropout,
            attn_dropout=attn_dropout,
        )

    def forward(self, x):
        x1 = self.transformer_encoder(x)
        x2 = self.transformer_encoder(x1)
        x3 = self.transformer_encoder(x2)
        x4 = self.transformer_encoder(x2, x3)
        x5 = self.transformer_encoder(x1, x4)
        x6 = self.transformer_encoder(x, x5)

        return x6
