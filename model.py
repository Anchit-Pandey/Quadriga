"""
model.py  ─  CS3T-UNet
Paper: "Cross-shaped Separated Spatial-Temporal UNet Transformer For
        Accurate Channel Prediction", IEEE INFOCOM 2024, Kang et al.

Paper section ↔ code mapping
  §II       System model – ADP input/output format
  §III-B    PatchEmbedding, overall encoder/decoder/skip layout
  §III-C1   CrossShapedSpatialAttention  (Eq. 5, 6)
            GroupWiseTemporalAttention   (Eq. 7, 8)
  §III-C2   CS3TBlock  (Fig. 8: LN→SpatialMSA→LN→TemporalMSA→LN→MLP)
  §III-D    MergeBlock (Fig. 7a), ExpandBlock (Fig. 7b), 4-level UNet (Fig. 3)
  §IV-A     C=64, blocks=(2,2,6,2), tanh output, AdamW lr=2e-3, batch=32, 400 ep

All assumptions are labeled [ASSUMPTION].
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
#  TEMPORAL POSITIONAL ENCODING   [Paper §III-C1, Eq. (7)]
#
#  "The positional encoding is added along the channel dimension C′."
#  PE_{pos,2i}   = sin(pos / 10000^{2i/C′})
#  PE_{pos,2i+1} = cos(pos / 10000^{2i/C′})
#
#  Channel dimension C = 2T encodes temporal information (T frames × real/imag).
#  The PE distinguishes which frame index each channel position corresponds to.
# ═══════════════════════════════════════════════════════════════════════════════
class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal PE broadcast over the channel (temporal) dim of (B, H, W, C)."""

    def __init__(self, max_len: int = 1024):
        super().__init__()
        pe  = torch.zeros(max_len)
        pos = torch.arange(max_len).float()
        denom = max_len + 1e-9
        pe[0::2] = torch.sin(pos[0::2] / (10000.0 ** (pos[0::2] / denom)))
        pe[1::2] = torch.cos(pos[1::2] / (10000.0 ** (pos[1::2] / denom)))
        self.register_buffer('pe', pe)   # (max_len,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C)  →  x + pe[:C] broadcast over B, H, W."""
        return x + self.pe[:x.size(-1)].view(1, 1, 1, -1)


# ═══════════════════════════════════════════════════════════════════════════════
#  CROSS-SHAPED SPATIAL SELF-ATTENTION   [Paper §III-C1, Eq. (5) & (6)]
#
#  "…performs self-attention in the horizontal and vertical stripes in parallel."
#  K heads split equally:
#    K/2 heads → horizontal stripe attention  (H-Attn)
#    K/2 heads → vertical   stripe attention  (V-Attn)
#  [Eq. 6]:  CSWin-Atten(X) = CONCAT[H-Attn₁…H-Attn_{K/2},
#                                      V-Attn_{K/2+1}…V-Attn_K]
#
#  Separate QKV projections per direction for clarity.
#  Stripe width sw: "we can enlarge the receptive field by adjusting sw"
# ═══════════════════════════════════════════════════════════════════════════════
class CrossShapedSpatialAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int = 8, stripe_width: int = 7,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert num_heads % 2 == 0, "num_heads must be even for cross-shaped attention"
        self.hn  = num_heads // 2   # heads per direction
        self.sw  = stripe_width     # stripe width  (sw in paper)
        self.hd  = dim // num_heads # head dimension
        self.Ch  = dim // 2         # channels per direction

        # Independent Q,K,V projections for horizontal and vertical branches
        self.qkv_h = nn.Linear(self.Ch, self.Ch * 3)
        self.qkv_v = nn.Linear(self.Ch, self.Ch * 3)
        self.proj  = nn.Linear(dim, dim)
        self.ad    = nn.Dropout(attn_drop)
        self.pd    = nn.Dropout(proj_drop)

    # ─── helpers ───────────────────────────────────────────────────────────
    def _mha(self, q, k, v):
        """q,k,v: (BM, L, hn, hd) → (BM, L, hn, hd)"""
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) * (self.hd ** -0.5)
        a = self.ad(a.softmax(dim=-1))
        return (a @ v).transpose(1, 2)

    # ─── main ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C)  →  (B, H, W, C)"""
        B, H, W, C = x.shape
        sw = self.sw; hn = self.hn; hd = self.hd; Ch = self.Ch

        # Split channels: first half → horizontal, second half → vertical
        xh = x[..., :Ch]
        xv = x[..., Ch:]

        # ── Horizontal stripe attention [Paper Eq. 5] ─────────────────────
        # "X evenly divided into M horizontal stripes, each with stripe width sw"
        ph = (sw - H % sw) % sw
        if ph: xh = F.pad(xh, (0, 0, 0, 0, 0, ph))
        Hp = xh.shape[1]; Mh = Hp // sw
        s  = xh.reshape(B, Mh, sw, W, Ch).reshape(B * Mh, sw * W, Ch)
        q, k, v = self.qkv_h(s).chunk(3, dim=-1)
        q = q.reshape(B * Mh, sw * W, hn, hd)
        k = k.reshape(B * Mh, sw * W, hn, hd)
        v = v.reshape(B * Mh, sw * W, hn, hd)
        oh = self._mha(q, k, v).reshape(B * Mh, sw * W, Ch).reshape(B, Hp, W, Ch)
        if ph: oh = oh[:, :H]

        # ── Vertical stripe attention ─────────────────────────────────────
        pw = (sw - W % sw) % sw
        if pw: xv = F.pad(xv, (0, 0, 0, pw))
        Wp = xv.shape[2]; Mv = Wp // sw
        # Transpose to (B, W, H, Ch) so we can stripe along the width dimension
        s  = xv.permute(0, 2, 1, 3).contiguous().reshape(B, Mv, sw, H, Ch).reshape(B * Mv, sw * H, Ch)
        q, k, v = self.qkv_v(s).chunk(3, dim=-1)
        q = q.reshape(B * Mv, sw * H, hn, hd)
        k = k.reshape(B * Mv, sw * H, hn, hd)
        v = v.reshape(B * Mv, sw * H, hn, hd)
        ov = self._mha(q, k, v).reshape(B * Mv, sw * H, Ch)
        ov = ov.reshape(B, Wp, H, Ch).permute(0, 2, 1, 3).contiguous()
        if pw: ov = ov[:, :, :W]

        # [Eq. 6] Concatenate horizontal and vertical outputs
        out = torch.cat([oh, ov], dim=-1)   # (B, H, W, C)
        return self.pd(self.proj(out))


# ═══════════════════════════════════════════════════════════════════════════════
#  GROUP-WISE TEMPORAL ATTENTION   [Paper §III-C1, Eq. (8)]
#
#  "We tokenize the channel dimension by evenly splitting into N non-overlapping
#   groups, each with sw channels. The MSA is performed within each temporal group."
#
#  Channel dim C = 2T encodes temporal information.
#  Splitting into groups of size sw allows efficient temporal feature capture.
#
#  [Paper §III-C2]: "The linear projection layer for temporal attention is
#   initialized as zero and then updated during training."
# ═══════════════════════════════════════════════════════════════════════════════
class GroupWiseTemporalAttention(nn.Module):

    def __init__(self, dim: int, group_size: int = 4, num_heads: int = 8,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.gs = group_size                 # sw  (sequence length per group)
        self.pe = TemporalPositionalEncoding()

        # QKV projects each sw-dimensional group token
        # [Paper §III-C2]: initialized to zero
        self.qkv  = nn.Linear(group_size, group_size * 3)
        nn.init.zeros_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)

        # Output projection over full channel dim — also zero-initialized
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        self.ad = nn.Dropout(attn_drop)
        self.pd = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C)  →  (B, H, W, C)"""
        B, H, W, C = x.shape
        sw = self.gs

        # Add temporal positional encoding [Paper Eq. 7]
        x = self.pe(x)

        # Pad C to multiple of sw
        pad = (sw - C % sw) % sw
        if pad: x = F.pad(x, (0, pad))
        Cp = x.shape[-1]; N = Cp // sw   # N groups

        # [Eq. 8]: X = [X₁, ..., X_N]; attention within each group X_i
        # Reshape to (B*H*W, N, sw): N groups as sequence, sw as feature dim
        xg = x.reshape(B * H * W, N, sw)
        q, k, v = self.qkv(xg).chunk(3, dim=-1)   # each (B*H*W, N, sw)

        # Scaled dot-product attention over N groups (scale by sw, the token dim)
        attn = (q @ k.transpose(-2, -1)) * (sw ** -0.5)
        attn = self.ad(attn.softmax(dim=-1))
        out  = (attn @ v).reshape(B, H, W, Cp)     # (B*H*W, N, sw) → (B,H,W,Cp)

        if pad: out = out[..., :C]
        return self.pd(self.proj(out))


# ═══════════════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK  (MLP in Fig. 8)
# ═══════════════════════════════════════════════════════════════════════════════
class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hd = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hd), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hd, dim), nn.Dropout(drop),
        )
    def forward(self, x): return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  CS3T BLOCK   [Paper §III-C, Fig. 8]
#
#  Input → LN → SpatialMSA ⊕ → LN → TemporalMSA ⊕ → LN → MLP ⊕ → Output
#  Residual connections (⊕) around every sub-module.
# ═══════════════════════════════════════════════════════════════════════════════
class CS3TBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int = 8,
                 stripe_width: int = 7, group_size: int = 4,
                 mlp_ratio: float = 4.0, attn_drop: float = 0.0, drop: float = 0.0):
        super().__init__()
        self.n1  = nn.LayerNorm(dim)
        self.n2  = nn.LayerNorm(dim)
        self.n3  = nn.LayerNorm(dim)
        self.sa  = CrossShapedSpatialAttention(dim, num_heads, stripe_width, attn_drop, drop)
        self.ta  = GroupWiseTemporalAttention(dim, group_size, num_heads, attn_drop, drop)
        self.ffn = FeedForward(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C)  →  (B, H, W, C)"""
        x = x + self.sa(self.n1(x))    # LN → Cross-shaped spatial MSA
        x = x + self.ta(self.n2(x))    # LN → Group-wise temporal MSA
        x = x + self.ffn(self.n3(x))   # LN → FFN
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  MERGE BLOCK   [Paper §III-C, Fig. 7a]
#  "Conv2D(kernel=3, stride=2, padding=1) → Norm"
#  Halves spatial dims, doubles channel dim.
# ═══════════════════════════════════════════════════════════════════════════════
class MergeBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, W, C) → (B, H/2, W/2, 2C)"""
        return self.norm(self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPAND BLOCK   [Paper §III-C, Fig. 7b]
#  "Linear(d_out=2C) → PixelShuffle → Norm"
#  Doubles spatial dims, halves channel dim.
#  Implementation: Linear(di → do*4) → PixelShuffle(2) → LayerNorm(do)
# ═══════════════════════════════════════════════════════════════════════════════
class ExpandBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        """dim_out = dim_in // 2  (spatial resolution doubled, channels halved)."""
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out * 4)
        self.norm   = nn.LayerNorm(dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, W, C) → (B, 2H, 2W, C/2)"""
        x = self.linear(x).permute(0, 3, 1, 2)   # (B, do*4, H, W)
        x = F.pixel_shuffle(x, 2)                  # (B, do,   2H, 2W)
        return self.norm(x.permute(0, 2, 3, 1))    # (B, 2H, 2W, do)


# ═══════════════════════════════════════════════════════════════════════════════
#  PATCH EMBEDDING   [Paper §III-B]
#  "shallow feature embedding uses a convolutional layer to tokenize the input.
#   Token embedding dimension is C; token size is 2×2."
# ═══════════════════════════════════════════════════════════════════════════════
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 2):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C_in, Nf, Nt) → (B, Nf/2, Nt/2, embed_dim)"""
        return self.norm(self.proj(x).permute(0, 2, 3, 1))


# ═══════════════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  =  N × CS3TBlock  +  MergeBlock
# ═══════════════════════════════════════════════════════════════════════════════
class EncoderLayer(nn.Module):
    def __init__(self, dim: int, num_blocks: int, num_heads: int,
                 stripe_width: int, group_size: int, merge_out_dim: int = None,
                 attn_drop: float = 0.0, drop: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            CS3TBlock(dim, num_heads, stripe_width, group_size,
                      attn_drop=attn_drop, drop=drop)
            for _ in range(num_blocks)
        ])
        self.merge = MergeBlock(dim, merge_out_dim) if merge_out_dim else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        skip = x                    # feature map saved for skip connection
        if self.merge:
            x = self.merge(x)
        return x, skip


# ═══════════════════════════════════════════════════════════════════════════════
#  DECODER LAYER  =  M × CS3TBlock  +  ExpandBlock
#  Skip connection fused by linear projection after concatenation.
#  [Paper §III-C]: "shortcut connections aggregate multi-resolution features"
# ═══════════════════════════════════════════════════════════════════════════════
class DecoderLayer(nn.Module):
    def __init__(self, dim: int, num_blocks: int, num_heads: int,
                 stripe_width: int, group_size: int, skip_dim: int,
                 expand_out_dim: int = None,
                 attn_drop: float = 0.0, drop: float = 0.0):
        super().__init__()
        self.fuse   = nn.Linear(dim + skip_dim, dim)
        self.blocks = nn.ModuleList([
            CS3TBlock(dim, num_heads, stripe_width, group_size,
                      attn_drop=attn_drop, drop=drop)
            for _ in range(num_blocks)
        ])
        self.expand = ExpandBlock(dim, expand_out_dim) if expand_out_dim else None

    def forward(self, x, skip):
        x = self.fuse(torch.cat([x, skip], dim=-1))
        for blk in self.blocks:
            x = blk(x)
        if self.expand:
            x = self.expand(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  CS3T-UNet  (Full Architecture)   [Paper Fig. 3, §III-D, §IV-A]
#
#  Input format  [Paper §III-B]:
#    "combine temporal dimension with complex dimension → Nf × Nt × 2T"
#    x: (B, 2T, Nf, Nt)   — T frames × {real, imag}
#
#  Encoder (4 levels, after patch embedding):
#    PatchEmbed  : (B, 2T, Nf, Nt)     → (B, Nf/2,  Nt/2,  C)
#    Level 1     : → skip₁ (Nf/2, C)   → merge → (Nf/4,  2C)
#    Level 2     : → skip₂ (Nf/4, 2C)  → merge → (Nf/8,  4C)
#    Level 3     : → skip₃ (Nf/8, 4C)  → merge → (Nf/16, 8C)
#    Level 4     : → skip₄ (Nf/16, 8C) [bottleneck, no merge]
#
#  Decoder (mirror, skip connections fused at each level):
#    Dec 1  : feat(8C) + skip₄(8C) → 8C → expand → (Nf/8,  4C)
#    Dec 2  : feat(4C) + skip₃(4C) → 4C → expand → (Nf/4,  2C)
#    Dec 3  : feat(2C) + skip₂(2C) → 2C → expand → (Nf/2,  C)
#    Dec 4  : feat(C)  + skip₁(C)  → C  [no expand, stays at patch grid]
#
#  Final:
#    PixelShuffle(2) : (Nf/2, Nt/2, C) → (Nf, Nt, C)
#    Linear          : C → 2L
#    Tanh            : [Paper §IV-A] "tanh attached to outputs of all models"
#
#  Output: (B, 2L, Nf, Nt)
#
#  Hyperparameters [Paper §IV-A]:
#    C = 64,  blocks = (2,2,6,2),  patch_size = 2
#    [ASSUMPTION] stripe_width=7 (CSwin Transformer default, paper cites [13])
#    [ASSUMPTION] group_size=4  (not specified; balances granularity and cost)
# ═══════════════════════════════════════════════════════════════════════════════
class CS3TUNet(nn.Module):

    def __init__(
        self,
        in_channels:  int   = 20,        # 2T  (T=10 frames × real+imag)
        out_channels: int   = 10,        # 2L  (L=5 frames × real+imag)
        embed_dim:    int   = 64,        # C   [Paper §IV-A]
        num_blocks:   tuple = (2,2,6,2), # (N1,N2,N3,N4)  [Paper §IV-A]
        num_heads:    int   = 8,
        stripe_width: int   = 7,         # [ASSUMPTION: CSwin default]
        group_size:   int   = 4,         # [ASSUMPTION]
        attn_drop:    float = 0.0,
        drop:         float = 0.0,
    ):
        super().__init__()
        C = embed_dim
        N1, N2, N3, N4 = num_blocks

        # Shared keyword args for every encoder/decoder layer
        blk_kw = dict(num_heads=num_heads, stripe_width=stripe_width,
                      group_size=group_size, attn_drop=attn_drop, drop=drop)

        # ── Shallow feature embedding ──────────────────────────────────────
        self.patch_embed = PatchEmbedding(in_channels, C, patch_size=2)

        # ── Encoder ───────────────────────────────────────────────────────
        self.enc1 = EncoderLayer(C,   N1, merge_out_dim=C*2,  **blk_kw)
        self.enc2 = EncoderLayer(C*2, N2, merge_out_dim=C*4,  **blk_kw)
        self.enc3 = EncoderLayer(C*4, N3, merge_out_dim=C*8,  **blk_kw)
        self.enc4 = EncoderLayer(C*8, N4, merge_out_dim=None, **blk_kw)  # bottleneck

        # ── Decoder ───────────────────────────────────────────────────────
        self.dec1 = DecoderLayer(C*8, N4, skip_dim=C*8, expand_out_dim=C*4, **blk_kw)
        self.dec2 = DecoderLayer(C*4, N3, skip_dim=C*4, expand_out_dim=C*2, **blk_kw)
        self.dec3 = DecoderLayer(C*2, N2, skip_dim=C*2, expand_out_dim=C,   **blk_kw)
        self.dec4 = DecoderLayer(C,   N1, skip_dim=C,   expand_out_dim=None,**blk_kw)

        # ── Final projection ───────────────────────────────────────────────
        # Upsample from patch grid (Nf/2, Nt/2) → full resolution (Nf, Nt)
        self.up_linear  = nn.Linear(C, C * 4)    # feeds pixel_shuffle(2)
        self.final_proj = nn.Linear(C, out_channels)
        self.output_act = nn.Tanh()               # [Paper §IV-A]

        self._init_weights()

    # ─── weight initialisation ────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Skip zero-initialized temporal attention projections
                if not (m.weight.data == 0).all():
                    nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ─── forward ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2T, Nf, Nt)  — real/imag-interleaved T historical frames
        Returns:
            (B, 2L, Nf, Nt)     — real/imag-interleaved L predicted frames
        """
        # ── Shallow feature embedding ──────────────────────────────────────
        f = self.patch_embed(x)              # (B, Nf/2, Nt/2, C)

        # ── Encoder ────────────────────────────────────────────────────────
        f, s1 = self.enc1(f)                 # s1: (B, Nf/2,  Nt/2,  C)
        f, s2 = self.enc2(f)                 # s2: (B, Nf/4,  Nt/4,  2C)
        f, s3 = self.enc3(f)                 # s3: (B, Nf/8,  Nt/8,  4C)
        f, s4 = self.enc4(f)                 # s4: (B, Nf/16, Nt/16, 8C) [bottleneck]

        # ── Decoder (skip connections fused at every level) ─────────────────
        f = self.dec1(f, s4)                 # (B, Nf/8,  Nt/8,  4C)
        f = self.dec2(f, s3)                 # (B, Nf/4,  Nt/4,  2C)
        f = self.dec3(f, s2)                 # (B, Nf/2,  Nt/2,  C)
        f = self.dec4(f, s1)                 # (B, Nf/2,  Nt/2,  C)

        # ── Upsample from patch grid to full spatial resolution ────────────
        f = self.up_linear(f).permute(0, 3, 1, 2)   # (B, 4C, Nf/2, Nt/2)
        f = F.pixel_shuffle(f, 2).permute(0, 2, 3, 1)  # (B, Nf, Nt, C)

        # ── Final prediction ───────────────────────────────────────────────
        out = self.final_proj(f).permute(0, 3, 1, 2)  # (B, 2L, Nf, Nt)
        return self.output_act(out)


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable,
            'total_M': total / 1e6, 'trainable_M': trainable / 1e6}


def build_model(T: int = 10, L: int = 5, **kwargs) -> CS3TUNet:
    """Factory matching paper §IV-A defaults (QuaDRiGa dataset)."""
    defaults = dict(embed_dim=64, num_blocks=(2,2,6,2),
                    num_heads=8, stripe_width=7, group_size=4)
    defaults.update(kwargs)
    return CS3TUNet(in_channels=2*T, out_channels=2*L, **defaults)


# ═══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time
    print("=" * 60)
    print("  CS3T-UNet Smoke Test")
    print("=" * 60)

    # Shape tests
    print("\nShape & tanh tests:")
    for B, T, L in [(1,10,5),(2,10,1),(4,10,5)]:
        m = build_model(T=T, L=L)
        x = torch.randn(B, 2*T, 64, 64)
        t0 = time.time()
        y  = m(x)
        ms = (time.time() - t0) * 1000
        assert y.shape == (B, 2*L, 64, 64), f"Shape {y.shape}"
        assert y.abs().max() <= 1.0 + 1e-4,  f"tanh {y.abs().max()}"
        p = count_parameters(m)
        print(f"  B={B} T={T} L={L}: {tuple(x.shape)} → {tuple(y.shape)} "
              f"| {p['total_M']:.2f}M params | max={y.abs().max():.4f} | {ms:.0f}ms")

    # Gradient flow
    print("\nGradient flow test:")
    m = build_model()
    x = torch.randn(2, 20, 64, 64)
    m(x).sum().backward()
    ng = sum(1 for p in m.parameters() if p.grad is not None)
    nt = sum(1 for _  in m.parameters())
    print(f"  Params with gradients: {ng}/{nt}")
    assert ng == nt, "Some parameters have no gradient!"

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("  Note: paper reports 19.64M params;")
    print("  reproduction is ~25M due to fuse layers in skip connections.")
    print("  [ASSUMPTION] Skip fuse implemented as Linear concat;")
    print("  paper may use add instead — see README for details.")
    print("=" * 60)
