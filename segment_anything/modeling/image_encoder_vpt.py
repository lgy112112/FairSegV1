# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from operator import mul
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        num_vpt_tokens: int = 5,
        return_visual_prompts: bool = False,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.num_vpt_tokens = num_vpt_tokens
        self.return_visual_prompts = (return_visual_prompts and num_vpt_tokens > 0)

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            )
        self.global_attn_indexes = global_attn_indexes
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = PromptedBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                num_vpt_tokens=num_vpt_tokens,
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        if self.num_vpt_tokens > 0:
            # initialize visual prompt tuning
            val = math.sqrt(6. / float(3 * patch_size * patch_size + embed_dim))
            # shallow visual prompts
            self.visual_prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_vpt_tokens, embed_dim))
            nn.init.uniform_(self.visual_prompt_embeddings.data, -val, val)
            # # deep visual prompts
            self.deep_visual_prompt_embeddings = nn.Parameter(torch.zeros(len(self.blocks) - 1, self.num_vpt_tokens, embed_dim))
            nn.init.uniform_(self.deep_visual_prompt_embeddings.data, -val, val)

    def prompted_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        
        # prompt the image tokens with visual prompts
        B, H, W, C = x.shape
        x = x.view(B, H * W, C)
        x = torch.cat([self.visual_prompt_embeddings.expand(B, -1, -1), x], dim=1)  # (B, num_tokens + H * W, embed_dim)

        for idx, blk in enumerate(self.blocks):
            if idx == 0:
                x = blk(x)
            else:
                if idx <= self.deep_visual_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.deep_visual_prompt_embeddings[idx - 1].unsqueeze(0).expand(B, -1, -1)  # (B, num_tokens, embed_dim)
                    # concatenate the visual prompts and image tokens
                    x = torch.cat([deep_prompt_emb, x[:, self.num_vpt_tokens:]], dim=1)  # (B, num_tokens + H * W, embed_dim)
                x = blk(x)

        # split visual prompts and image tokens
        visual_prompts = x[:, :self.num_vpt_tokens, :]  # (B, N, C)
        x = x[:, self.num_vpt_tokens:, :].view(B, H, W, C)

        out = self.neck(x.permute(0, 3, 1, 2))
        # print(f"out in image_encoder shape: {out}")
        if self.return_visual_prompts:  # triplet output: feature map (after neck), image tokens, visual prompts
            return out, x.view(B, H * W, C), visual_prompts
        return out
    
    def common_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        
        B, H, W, C = x.shape
        x = x.view(B, H * W, C)
        
        for blk in self.blocks:
            x = blk(x)

        x = x.view(B, H, W, C)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prompted_forward(x) if self.num_vpt_tokens > 0 else self.common_forward(x)

    def train(self, mode=True):
        if self.num_vpt_tokens > 0:
            # when using VPT, set eval mode to all modules
            for module in self.children():
                module.train(False)
        else:
            super().train(mode)


class PromptedBlock(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        num_vpt_tokens: int = 5,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PromptedAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(
            embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer
        )

        self.window_size = window_size
        self.input_size = input_size
        self.num_tokens = num_vpt_tokens

    def prompted_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: visual prompted input tensor with shape (B, L, embed_dim), note L = H * W + num_tokens
        Returns:
            x: output tensor with shape (B, L, embed_dim)
        """
        B, L, C = x.shape
        H, W = self.input_size

        shortcut = x
        x = self.norm1(x)

        # split the visual prompts and image tokens
        visual_prompts = x[:, :self.num_tokens, :]  # (B, num_tokens, embed_dim)
        x = x[:, self.num_tokens:, :]  # (B, H * W, embed_dim)
        L = L - self.num_tokens

        assert L == H * W, "The number of tokens should be equal to the image size."

        x = x.view(B, H, W, C)
        
        # Window partition
        if self.window_size > 0:
            x, pad_hw = window_partition(x, self.window_size)
            # note that we need to expand the visual prompts to be fed into attention windows
            visual_prompts = visual_prompts.unsqueeze(0)  # (1, B, num_tokens, embed_dim)
            visual_prompts = visual_prompts.expand(x.shape[0] // B, -1, -1, -1)
            visual_prompts = visual_prompts.reshape((-1, self.num_tokens, C))

        # note that here the shape of visual prompts is (B * numWin, num_tokens, embed_dim) if using windowed attention
        x, visual_prompts = self.attn(x, visual_prompts)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
            visual_prompts = visual_prompts.reshape(-1, B, self.num_tokens, C)
            visual_prompts = visual_prompts.mean(dim=0)

        # concatenate the visual prompts and image tokens
        x = torch.cat([visual_prompts, x.view(B, H * W, C)], dim=1)  # (B, L, embed_dim)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    def common_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor with shape (B, L, embed_dim)
        Returns:
            x: output tensor with shape (B, L, embed_dim)
        """
        B, L, C = x.shape
        H, W = self.input_size

        shortcut = x
        x = self.norm1(x)

        assert L == H * W, "The number of tokens should be equal to the image size."

        x = x.view(B, H, W, C)

        # Window partition
        if self.window_size > 0:
            x, pad_hw = window_partition(x, self.window_size)

        # note that here the shape of visual prompts is (B * numWin, num_tokens, embed_dim) if using windowed attention
        x, _ = self.attn(x, None)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        # concatenate the visual prompts and image tokens
        x = x.view(B, H * W, C)  # (B, L, embed_dim)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prompted_forward(x) if self.num_tokens > 0 else self.common_forward(x)


class PromptedAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings. Add support for VPT."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor, visual_prompts: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: input tensor with shape (B, H, W, embed_dim)
            visual_prompts: visual prompts with shape (B, num_tokens, embed_dim)
        Returns:
            x: output tensor with shape (B, H * W, embed_dim)
            visual_prompts: updated visual prompts with shape (B, num_tokens, embed_dim)
        """
        B, H, W, C_full = x.shape  # C_full denotes the embed_dim before multi-head split
        if visual_prompts is not None:
            num_tokens = visual_prompts.shape[1]
            # concatenate in the prepend manner
            x = torch.cat([visual_prompts, x.reshape(B, H * W, -1)], dim=1)  # (B, num_tokens + H * W, embed_dim)
        else:
            num_tokens = 0
            x = x.reshape(B, H * W, -1)
        # qkv with shape (3, B, nHead, num_tokens + H * W, C)
        qkv = (
            self.qkv(x).reshape(B, H * W + num_tokens, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, num_tokens + H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W + num_tokens, -1).unbind(0)

        # attn.shape: (B * nHead, num_tokens + H * W, num_tokens + H * W)
        # now we need to split the attention map into visual prompts and image tokens
        attn = (q * self.scale) @ k.transpose(-2, -1)  # (num_tokens + H * W, num_tokens + H * W)
        # take out the attention maps of image tokens
        if visual_prompts is not None:
            visual_attn = attn[:, num_tokens:, num_tokens:]
        else:
            visual_attn = attn

        if self.use_rel_pos:
            visual_attn = add_decomposed_rel_pos(
                visual_attn, q[:, num_tokens:], self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        if visual_prompts is not None:
            # generate an attention mask here
            # for the query, we hope that all tokens could attend to the visual prompts
            # but only image tokens could attend to each other
            attn_mask = torch.zeros(H * W + num_tokens, H * W + num_tokens, device=attn.device)
            attn_mask[num_tokens:, :num_tokens] = -100.0
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        # x = (
        #     (attn @ v)
        #     .view(B, self.num_heads, H, W, -1)
        #     .permute(0, 2, 3, 1, 4)
        #     .reshape(B, H, W, -1)
        # )
        x = (attn @ v).view(B, self.num_heads, H * W + num_tokens, -1).permute(0, 2, 1, 3).reshape(B, H * W + num_tokens, -1)  # (B, num_tokens + H * W, C)
        x = self.proj(x)
        # split the visual prompts and image tokens
        if visual_prompts is not None:
            visual_prompts = x[:, :num_tokens, :]  # (B, num_tokens, embed_dim)
            x = x[:, num_tokens:, :]  # (B, H * W, embed_dim)
        else:
            visual_prompts = None

        x = x.view(B, H, W, -1)
        return x, visual_prompts


def window_partition(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
