# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from .attention import flash_attention
from torch.utils.checkpoint import checkpoint
from foley_omni.distributed_comms.communications import all_gather, all_to_all_4D
from foley_omni.distributed_comms.parallel_states import nccl_info, get_sequence_parallel_state


def gradient_checkpointing(module: nn.Module, *args, enabled: bool, **kwargs):
    if enabled:
        return checkpoint(module, *args, use_reentrant=False, **kwargs)
    else:
        return module(*args, **kwargs)


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000, freqs_scaling=1.0):
    assert dim % 2 == 0
    pos =  torch.arange(max_seq_len)
    freqs = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    freqs = freqs_scaling * freqs
    freqs = torch.outer(pos, freqs)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

@amp.autocast('cuda', enabled=False)
def rope_freqs_at_positions(positions, dim, theta=10000, freqs_scaling=1.0):
    assert dim % 2 == 0
    positions = positions.to(torch.float64)
    freqs = 1.0 / torch.pow(
        theta,
        torch.arange(0, dim, 2, device=positions.device).to(torch.float64).div(dim))
    freqs = freqs_scaling * freqs
    angles = torch.outer(positions, freqs)
    freqs = torch.polar(torch.ones_like(angles), angles)
    return freqs

@amp.autocast('cuda', enabled=False)
def rope_apply_1d(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2 ## b l h d
    c_rope = freqs.shape[1]  # number of complex dims to rotate
    assert c_rope <= c, "RoPE dimensions cannot exceed half of hidden size"
    
    # loop over samples
    output = []
    for i, (l, ) in enumerate(grid_sizes.tolist()):
        seq_len = l
        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2)) # [l n d//2]
        x_i_rope = x_i[:, :, :c_rope] * freqs[:seq_len, None, :]  # [L, N, c_rope]
        x_i_passthrough = x_i[:, :, c_rope:]  # untouched dims
        x_i = torch.cat([x_i_rope, x_i_passthrough], dim=2)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).bfloat16()

@amp.autocast('cuda', enabled=False)
def rope_apply_3d(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    
    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).bfloat16()

@amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    x_ndim = grid_sizes.shape[-1]
    if x_ndim == 3:
        return rope_apply_3d(x, grid_sizes, freqs)
    else:
        return rope_apply_1d(x, grid_sizes, freqs)


@amp.autocast('cuda', enabled=False)
def rope_apply_tokens_1d(x, freqs):
    """
    Apply 1D RoPE on token embeddings.

    Args:
        x(Tensor): [B, L, D]
        freqs(Tensor): complex tensor [L, D_rope/2]
    """
    assert x.ndim == 3, f"Expected [B, L, D], got {x.shape}"
    b, l, d = x.shape
    if l == 0 or d < 2:
        return x

    # RoPE works on pairs; keep odd tail dim untouched.
    d_even = (d // 2) * 2
    c = d_even // 2
    c_rope = min(c, freqs.shape[1])
    if c_rope <= 0:
        return x

    x_dtype = x.dtype
    x_main = x[..., :d_even].to(torch.float32).reshape(b, l, c, 2)
    x_main = torch.view_as_complex(x_main)  # [B, L, C]

    rope_freqs = freqs[:l, :c_rope].to(x.device)
    x_rope = x_main[:, :, :c_rope] * rope_freqs.unsqueeze(0)
    x_main = torch.cat([x_rope, x_main[:, :, c_rope:]], dim=-1)

    x_main = torch.view_as_real(x_main).reshape(b, l, d_even)
    if d_even < d:
        x_out = torch.cat([x_main, x[..., d_even:].to(torch.float32)], dim=-1)
    else:
        x_out = x_main
    return x_out.to(x_dtype)

class ChannelLastConv1d(nn.Conv1d):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x


class ConvMLP(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ChannelLastConv1d(dim,
                                    hidden_dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)
        self.w2 = ChannelLastConv1d(hidden_dim,
                                    dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)
        self.w3 = ChannelLastConv1d(dim,
                                    hidden_dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SyncTemporalAdapter(nn.Module):

    def __init__(self,
                 dim: int,
                 adapter_type: str = "depthwise_conv",
                 kernel_size: int = 5,
                 expansion: float = 2.0):
        super().__init__()
        assert kernel_size % 2 == 1, "sync adapter kernel size must be odd"
        assert expansion >= 1.0, "sync adapter expansion must be >= 1.0"

        padding = kernel_size // 2
        hidden_dim = max(dim, int(dim * expansion))
        self.adapter_type = adapter_type
        self.norm = nn.LayerNorm(dim)

        if adapter_type == "depthwise_conv":
            self.net = nn.Sequential(
                ChannelLastConv1d(dim,
                                  dim,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  groups=dim,
                                  bias=False),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            out_proj = self.net[-1]
        elif adapter_type == "glu_conv":
            self.net = nn.Sequential(
                ChannelLastConv1d(dim, dim * 2, kernel_size=kernel_size, padding=padding),
                nn.GLU(dim=-1),
                nn.Linear(dim, dim),
            )
            out_proj = self.net[-1]
        elif adapter_type == "temporal_mlp":
            self.net = nn.Sequential(
                ChannelLastConv1d(dim, hidden_dim, kernel_size=kernel_size, padding=padding),
                nn.SiLU(),
                nn.Linear(hidden_dim, dim),
            )
            out_proj = self.net[-1]
        else:
            raise ValueError(f"Unsupported sync adapter type: {adapter_type}")

        # Zero-init the residual branch so enabling the adapter preserves the old path at step 0.
        nn.init.zeros_(out_proj.weight)
        if out_proj.bias is not None:
            nn.init.zeros_(out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.net(x)
        return residual + x


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.bfloat16()).type_as(x) * self.weight.bfloat16()

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.bfloat16()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        # optional sequence parallelism
        # self.world_size = get_world_size()
        self.use_sp = get_sequence_parallel_state()
        if self.use_sp:
            self.sp_size = nccl_info.sp_size
            self.sp_rank = nccl_info.rank_within_group
            assert self.num_heads % self.sp_size == 0, \
                f"Num heads {self.num_heads} must be divisible by sp_size {self.sp_size}"
    # query, key, value function
    def qkv_fn(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        q, k, v = self.qkv_fn(x)
        if self.use_sp:
            # print(f"[DEBUG SP] Doing all to all to shard head")
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = all_to_all_4D(k, scatter_dim=2, gather_dim=1)
            v = all_to_all_4D(v, scatter_dim=2, gather_dim=1) # [B, L, H/P, C/H]
        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)
        if self.use_sp: 
            # print(f"[DEBUG SP] Doing all to all to shard sequence")
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2) # [B, L/P, H, C/H]
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def qkv_fn(self, x, context):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        return q, k, v

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        q, k, v = self.qkv_fn(x, context)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 additional_emb_length=None):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.additional_emb_length = additional_emb_length

    def qkv_fn(self, x, context):
        context_img = context[:, : self.additional_emb_length]
        context = context[:, self.additional_emb_length :]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)

        return q, k, v, k_img, v_img


    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        q, k, v, k_img, v_img = self.qkv_fn(x, context)

        if self.use_sp:
            # print(f"[DEBUG SP] Doing all to all to shard head")
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)  
            k = torch.chunk(k, self.sp_size, dim=2)[self.sp_rank]
            v = torch.chunk(v, self.sp_size, dim=2)[self.sp_rank]
            k_img = torch.chunk(k_img, self.sp_size, dim=2)[self.sp_rank]
            v_img = torch.chunk(v_img, self.sp_size, dim=2)[self.sp_rank]
            
        # [B, L, H/P, C/H]
        # k_img: [B, L, H, C/H]
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)
        if self.use_sp: 
            # print(f"[DEBUG SP] Doing all to all to shard sequence")
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2) # [B, L/P, H, C/H]
            
        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}

class ModulationAdd(nn.Module):
    def __init__(self, dim, num):
        super().__init__()
        self.modulation = nn.Parameter(torch.randn(1, num, dim) / dim**0.5)

    def forward(self, e):
        return self.modulation.bfloat16() + e.bfloat16()

class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 additional_emb_length=None,
                 use_sync_adaln=False):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_sync_adaln = use_sync_adaln

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        if cross_attn_type == 'i2v_cross_attn':
            assert additional_emb_length is not None, "additional_emb_length should be specified for i2v_cross_attn"
            self.cross_attn = WanI2VCrossAttention(dim,
                                                num_heads,
                                                (-1, -1),
                                                qk_norm,
                                                eps, 
                                                additional_emb_length)
        else:
            assert additional_emb_length is None, "additional_emb_length should be None for t2v_cross_attn"
            self.cross_attn = WanT2VCrossAttention(dim,
                                                num_heads,
                                                (-1, -1),
                                                qk_norm,
                                                eps, )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        # self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        # self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.modulation = ModulationAdd(dim, 6)
        self.sync_adaln = (nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
                           if use_sync_adaln else None)


    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        sync_aligned=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.bfloat16
        assert len(e.shape) == 4 and e.size(2) == 6 and e.shape[1] == x.shape[1], f"{e.shape}, {x.shape}"

        # Optional per-block sync injection for audio branch.
        if sync_aligned is not None:
            x = x + sync_aligned

        with amp.autocast('cuda', dtype=torch.bfloat16):
            e = self.modulation(e)
            if self.sync_adaln is not None and sync_aligned is not None:
                sync_mod = self.sync_adaln(sync_aligned).unflatten(2, (6, self.dim))
                e = e + sync_mod.to(e.dtype)
            e = e.chunk(6, dim=2)
        assert e[0].dtype == torch.bfloat16

        # self-attention
        y = self.self_attn(
            self.norm1(x).bfloat16() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs)
        with amp.autocast('cuda', dtype=torch.bfloat16):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(
                self.norm2(x).bfloat16() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with amp.autocast('cuda', dtype=torch.bfloat16):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L, C]
        """
        assert e.dtype == torch.bfloat16
        with amp.autocast('cuda', dtype=torch.bfloat16):
            e = (self.modulation.bfloat16().unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2) # 1 1 2 D, B L 1 D -> B L 2 D -> 2 * (B L 1 D)
            x = (self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x



class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting audio generation in the public release, with legacy multimodal variants retained for compatibility.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size',
        'audio_fps', 'clip_fps', 'sync_fps'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2a',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 additional_emb_dim=None,
                 additional_emb_length=None,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 gradient_checkpointing = False,
                 temporal_rope_scaling_factor=1.0,
                 clip_len: int = 80,
                 sync_len: int = 240,
                 sync_add_to_audio: bool = True,
                 sync_add_to_audio_each_block: bool = False,
                 sync_use_adaln: bool = False,
                 sync_drop_from_context: bool = True,
                 sync_align_mode: str = "nearest-exact",
                 sync_use_proj: bool = False,
                 sync_use_adapter: bool = False,
                 sync_adapter_type: str = "depthwise_conv",
                 sync_adapter_kernel_size: int = 5,
                 sync_adapter_expansion: float = 2.0,
                 use_continuous_time_context_rope: bool = False,
                 continuous_time_context_rope_mode: str = "auto",
                 use_text_context_rope: bool = True,
                 use_clip_context_rope: bool = True,
                 use_sync_context_rope: bool = True,
                 context_rope_theta: float = 10000.0,
                 context_text_rope_scaling: float = 1.0,
                 context_clip_rope_scaling: float = 1.0,
                 context_sync_rope_scaling: float = 1.0,
                 audio_fps=None,
                 clip_fps=None,
                 sync_fps=None,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2a'):
                Model variant. The public release uses audio variants such as 't2a' and 'tt2a'. Legacy video variants are retained for checkpoint compatibility
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            clip_len (`int`, *optional*, defaults to 80):
                Fixed length for CLIP embeddings (pad with zeros or truncate)
            sync_len (`int`, *optional*, defaults to 240):
                Fixed length for Sync embeddings (pad with zeros or truncate)
            sync_add_to_audio (`bool`, *optional*, defaults to True):
                Whether to add aligned sync features to audio tokens
            sync_drop_from_context (`bool`, *optional*, defaults to True):
                Whether to remove sync features from cross-attn context
            sync_align_mode (`str`, *optional*, defaults to "nearest-exact"):
                Interpolation mode for sync alignment
            sync_use_proj (`bool`, *optional*, defaults to False):
                Whether to apply a linear projection before injection
            sync_use_adapter (`bool`, *optional*, defaults to False):
                Whether to apply a lightweight temporal adapter before sync injection
            use_continuous_time_context_rope (`bool`, *optional*, defaults to False):
                Whether to map CLIP/Sync positions onto the audio latent time axis before applying context RoPE
            continuous_time_context_rope_mode (`str`, *optional*, defaults to "auto"):
                "auto" uses runtime sequence-density scaling (audio_seq_len / context_len), while "fps" uses explicit audio/source fps
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 't2a', 'tt2a', 'ti2v'] ## tt2a means text transcript + text description to audio (to support both TTS and T2A
        self.model_type = model_type
        is_audio_type = "a" in self.model_type
        is_video_type = "v" in self.model_type
        assert is_audio_type ^ is_video_type, "Either audio or video model should be specified"
        if is_audio_type:
            ## audio model
            assert len(patch_size) == 1 and patch_size[0] == 1, "Audio model should only accept 1 dimensional input, and we dont do patchify"

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.temporal_rope_scaling_factor = temporal_rope_scaling_factor
        self.audio_fps = float(audio_fps) if audio_fps is not None else None
        self.clip_fps = float(clip_fps) if clip_fps is not None else None
        self.sync_fps = float(sync_fps) if sync_fps is not None else None
        self.clip_len = clip_len if is_audio_type else None
        self.sync_len = sync_len if is_audio_type else None
        self.is_audio_type = is_audio_type
        self.is_video_type = is_video_type
        # embeddings
        if is_audio_type:
            ## hardcoded to MMAudio
            self.patch_embedding = nn.Sequential(
                ChannelLastConv1d(in_dim, dim, kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(dim, dim * 4, kernel_size=7, padding=3),
            )
        else:
            self.patch_embedding = nn.Conv3d(
                in_dim, dim, kernel_size=patch_size, stride=patch_size)
            
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        # CLIP 和 Sync 特征的 embedding 层
        # CLIP 特征维度: 1024, Sync 特征维度: 768
        self.clip_emb = MLPProj(1024, dim) if is_audio_type else None
        self.sync_emb = MLPProj(768, dim) if is_audio_type else None

        # Scheme A controls
        self.sync_add_to_audio = sync_add_to_audio
        self.sync_add_to_audio_each_block = sync_add_to_audio_each_block
        self.sync_use_adaln = sync_use_adaln
        self.sync_drop_from_context = sync_drop_from_context
        self.sync_align_mode = sync_align_mode
        self.sync_use_proj = sync_use_proj
        self.sync_proj = nn.Linear(dim, dim) if (is_audio_type and sync_use_proj) else None
        self.sync_use_adapter = bool(sync_use_adapter)
        self.sync_adapter_type = str(sync_adapter_type)
        self.sync_adapter_kernel_size = int(sync_adapter_kernel_size)
        self.sync_adapter_expansion = float(sync_adapter_expansion)
        self.sync_adapter = (
            SyncTemporalAdapter(
                dim,
                adapter_type=self.sync_adapter_type,
                kernel_size=self.sync_adapter_kernel_size,
                expansion=self.sync_adapter_expansion,
            ) if (is_audio_type and self.sync_use_adapter) else None
        )
        self.use_continuous_time_context_rope = bool(use_continuous_time_context_rope)
        self.continuous_time_context_rope_mode = str(continuous_time_context_rope_mode)
        if self.continuous_time_context_rope_mode not in {"auto", "fps"}:
            raise ValueError(
                f"Unsupported continuous_time_context_rope_mode: {self.continuous_time_context_rope_mode}")
        self.use_text_context_rope = bool(use_text_context_rope)
        self.use_clip_context_rope = bool(use_clip_context_rope)
        self.use_sync_context_rope = bool(use_sync_context_rope)
        self.context_rope_theta = float(context_rope_theta)
        self.context_text_rope_scaling = float(context_text_rope_scaling)
        self.context_clip_rope_scaling = float(context_clip_rope_scaling)
        self.context_sync_rope_scaling = float(context_sync_rope_scaling)
        self.text_context_freqs = None
        self.clip_context_freqs = None
        self.sync_context_freqs = None

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.use_sp = get_sequence_parallel_state() # seq parallel
        if self.use_sp:
            self.sp_size = nccl_info.sp_size
            self.sp_rank = nccl_info.rank_within_group
            assert self.num_heads % self.sp_size == 0, \
                f"Num heads {self.num_heads} must be divisible by sp_size {self.sp_size}"
        # blocks
        # Legacy compatibility: i2v and tt2a share one cross-attention path, while t2v and t2a share the other.
        cross_attn_type = 't2v_cross_attn' if model_type in ['t2v', 't2a', 'ti2v'] else 'i2v_cross_attn'

        if cross_attn_type == 't2v_cross_attn':
            assert additional_emb_dim is None and additional_emb_length is None, "additional_emb_length should be None for t2v and t2a model"
        else:
            assert additional_emb_dim is not None and additional_emb_length is not None, "additional_emb_length should be specified for i2v and tt2a model"

        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps, additional_emb_length,
                              use_sync_adaln=(is_audio_type and sync_use_adaln))
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        self.set_gradient_checkpointing(enable=gradient_checkpointing)
        self.set_rope_params()
        self.set_context_rope_params()

        if model_type in ['i2v', 'tt2a']:
            self.img_emb = MLPProj(additional_emb_dim, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

    def set_rope_params(self):
        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        dim = self.dim
        num_heads = self.num_heads
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads

        if self.is_audio_type:
            ## to be determined
            # self.freqs = rope_params(1024, d, freqs_scaling=temporal_rope_scaling_factor)
            self.freqs = rope_params(1024, d - 4 * (d // 6), freqs_scaling=self.temporal_rope_scaling_factor)
        else:
            self.freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
                                dim=1)

    def set_context_rope_params(self):
        self.text_context_freqs = None
        self.clip_context_freqs = None
        self.sync_context_freqs = None
        if not self.is_audio_type:
            return

        if self.use_text_context_rope and self.text_len is not None and self.text_len > 0:
            self.text_context_freqs = rope_params(
                self.text_len,
                self.dim,
                theta=self.context_rope_theta,
                freqs_scaling=self.context_text_rope_scaling,
            )
        if self.use_clip_context_rope and self.clip_len is not None and self.clip_len > 0:
            self.clip_context_freqs = rope_params(
                self.clip_len,
                self.dim,
                theta=self.context_rope_theta,
                freqs_scaling=self.context_clip_rope_scaling,
            )
        if self.use_sync_context_rope and self.sync_len is not None and self.sync_len > 0:
            self.sync_context_freqs = rope_params(
                self.sync_len,
                self.dim,
                theta=self.context_rope_theta,
                freqs_scaling=self.context_sync_rope_scaling,
            )

    def _get_context_rope_freqs(self, name: str, length: int, scaling: float, device: torch.device):
        """
        Lazily build/refresh context RoPE freqs on real device.
        This avoids `.to()` on meta tensors (common with lazy/meta init).
        """
        freqs = getattr(self, name, None)
        need_rebuild = (
            freqs is None
            or getattr(freqs, "is_meta", False)
            or freqs.device != device
            or freqs.shape[0] < length
        )
        if need_rebuild:
            freqs = rope_params(
                length,
                self.dim,
                theta=self.context_rope_theta,
                freqs_scaling=scaling,
            ).to(device)
            setattr(self, name, freqs)
        return freqs

    def _build_continuous_context_rope_freqs(self,
                                             length: int,
                                             target_length: int,
                                             scaling: float,
                                             device: torch.device,
                                             source_fps: float = None):
        if (not self.use_continuous_time_context_rope or length <= 0 or target_length is None
                or target_length <= 0):
            return None

        if self.continuous_time_context_rope_mode == "fps":
            if (self.audio_fps is None or source_fps is None or self.audio_fps <= 0 or source_fps <= 0):
                return None
            position_step = self.audio_fps / source_fps
        else:
            # Match the original MMAudio idea: align context RoPE pace to the audio token density.
            position_step = float(target_length) / float(length)

        positions = torch.arange(length, device=device, dtype=torch.float64) * position_step
        return rope_freqs_at_positions(
            positions,
            self.dim,
            theta=self.context_rope_theta,
            freqs_scaling=scaling,
        )


    def set_gradient_checkpointing(self, enable: bool):
        self.gradient_checkpointing = enable
    
    def _set_gradient_checkpointing(self, module, value=False):
        """For diffusers compatibility"""
        self.gradient_checkpointing = value
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing - diffusers compatible"""
        self.gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing"""
        self.gradient_checkpointing = False

    def _align_sync_to_audio(self, sync_features, seq_lens, seq_len, device, dtype):
        if sync_features is None:
            return None
        if self.sync_emb is None:
            return None

        batch_size = seq_lens.shape[0]
        aligned_list = []

        for i in range(batch_size):
            if i >= len(sync_features) or sync_features[i] is None:
                aligned = torch.zeros(seq_len, self.dim, device=device, dtype=dtype)
                aligned_list.append(aligned)
                continue

            sync_feat = sync_features[i].to(device=device, dtype=dtype)
            if sync_feat.ndim == 3:
                sync_feat = sync_feat.squeeze(0)  # (T_sync, 768)

            if sync_feat.ndim != 2 or sync_feat.shape[-1] != 768:
                aligned = torch.zeros(seq_len, self.dim, device=device, dtype=dtype)
                aligned_list.append(aligned)
                continue

            # (T_sync, dim)
            sync_emb = self.sync_emb(sync_feat)

            # align to L_audio
            target_len = int(seq_lens[i].item())
            if target_len <= 0:
                aligned = torch.zeros(seq_len, self.dim, device=device, dtype=dtype)
                aligned_list.append(aligned)
                continue

            sync_emb = sync_emb.transpose(0, 1).unsqueeze(0)  # (1, dim, T_sync)
            sync_emb = F.interpolate(
                sync_emb.float(),
                size=target_len,
                mode=self.sync_align_mode,
            ).to(dtype)
            sync_emb = sync_emb.squeeze(0).transpose(0, 1)  # (L_audio, dim)

            if self.sync_use_proj and self.sync_proj is not None:
                sync_emb = self.sync_proj(sync_emb)

            if self.sync_adapter is not None:
                sync_emb = self.sync_adapter(sync_emb.unsqueeze(0)).squeeze(0)

            if target_len < seq_len:
                pad = torch.zeros(seq_len - target_len, self.dim, device=device, dtype=dtype)
                sync_emb = torch.cat([sync_emb, pad], dim=0)

            aligned_list.append(sync_emb)

        return torch.stack(aligned_list, dim=0)  # (B, seq_len, dim)

    def prepare_transformer_block_kwargs(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        clip_features=None,
        sync_features=None,
        y=None,
        first_frame_is_clean=False,
    ):

        # params
        ## need to change!
        device = x[0].device

        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x] ## x is list of [B L D] or [B C F H W]
        if self.is_audio_type:
            # [B, 1]
            grid_sizes = torch.stack(
                [torch.tensor(u.shape[1:2], dtype=torch.long) for u in x]
            )
        else:
            # [B, 3]
            grid_sizes = torch.stack(
                [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x] # [B C F H W] -> [B (F H W) C] -> [B L C]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len, f"Sequence length {seq_lens.max()} exceeds maximum {seq_len}."
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ]) # single [B, L, C]

        # time embeddings
        if t.dim() == 1:
            if first_frame_is_clean:
                t = torch.ones((t.size(0), seq_len), device=t.device, dtype=t.dtype) * t.unsqueeze(1)
                _first_images_seq_len = grid_sizes[:, 1:].prod(-1)
                for i in range(t.size(0)):
                    t[i, :_first_images_seq_len[i]] = 0
                # print(f"zeroing out first {_first_images_seq_len} from t: {t.shape}, {t}")
            else:
                t = t.unsqueeze(1).expand(t.size(0), seq_len)
        with amp.autocast('cuda', dtype=torch.bfloat16):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).bfloat16())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim)) # [1, 26784, 6, 3072] - B, seq_len, 6, dim
            assert e.dtype == torch.bfloat16 and e0.dtype == torch.bfloat16

        
        if self.use_sp:
            current_len = x.shape[1]
            # we will pad up to the next multiple of sp_size: eg. [157] -> [160]
            pad_size = (-current_len ) % self.sp_size  

            if pad_size > 0:
                padding = torch.zeros(
                    x.shape[0], pad_size, x.shape[2],
                    device=x.device,
                    dtype=x.dtype
                )
                x = torch.cat([x, padding], dim=1)
                e_padding = torch.zeros(
                    e.shape[0], pad_size, e.shape[2],
                    device=e.device,
                    dtype=e.dtype
                )
                e = torch.cat([e, e_padding], dim=1)
                e0_padding = torch.zeros(
                    e0.shape[0], pad_size, e0.shape[2], e0.shape[3],
                    device=e0.device,
                    dtype=e0.dtype
                )
                e0 = torch.cat([e0, e0_padding], dim=1)

            x = torch.chunk(x, self.sp_size, dim=1)[self.sp_rank]
            e = torch.chunk(e, self.sp_size, dim=1)[self.sp_rank]
            e0 = torch.chunk(e0, self.sp_size, dim=1)[self.sp_rank] 
            
        # context - 先处理 text embedding
        text_context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))  # [B, text_len, dim]
        if self.is_audio_type and self.use_text_context_rope and self.text_context_freqs is not None:
            text_freqs = self._get_context_rope_freqs(
                name="text_context_freqs",
                length=text_context.shape[1],
                scaling=self.context_text_rope_scaling,
                device=text_context.device,
            )
            text_context = rope_apply_tokens_1d(text_context, text_freqs)

        # 处理 CLIP、Sync 特征（仅在 audio 模式下）：与 text 一起固定长度 concat，不足补零，超过截断
        # context = [text | clip | sync]
        if self.is_audio_type:
            batch_size = text_context.shape[0]
            device = text_context.device
            dtype = text_context.dtype
            # 固定总长度 = text_len + clip_len + sync_len
            max_context_len = self.text_len + self.clip_len + self.sync_len
            video_contexts = []

            for i in range(batch_size):
                target_audio_len = int(seq_lens[i].item())
                # text 部分（已固定 text_len）
                context_parts = [text_context[i]]  # (text_len, dim)

                # CLIP 特征：不足补零，超过截断到 clip_len；无 CLIP 时补零
                if self.clip_emb is not None:
                    if clip_features is not None and i < len(clip_features) and clip_features[i] is not None:
                        clip_feat = clip_features[i].to(device=device, dtype=dtype)
                        if clip_feat.ndim == 3:
                            clip_feat = clip_feat.squeeze(0)  # (T_clip, 1024)
                        if clip_feat.ndim == 2 and clip_feat.shape[-1] == 1024:
                            T = clip_feat.size(0)
                            if T < self.clip_len:
                                pad = torch.zeros(self.clip_len - T, 1024, device=device, dtype=dtype)
                                clip_feat = torch.cat([clip_feat, pad], dim=0)
                            else:
                                clip_feat = clip_feat[:self.clip_len]
                            clip_feat_emb = self.clip_emb(clip_feat)  # (clip_len, dim)
                            if self.use_clip_context_rope and self.clip_context_freqs is not None:
                                clip_freqs = self._build_continuous_context_rope_freqs(
                                    clip_feat_emb.shape[0],
                                    target_audio_len,
                                    self.context_clip_rope_scaling,
                                    device,
                                    source_fps=self.clip_fps,
                                )
                                if clip_freqs is None:
                                    clip_freqs = self._get_context_rope_freqs(
                                        name="clip_context_freqs",
                                        length=clip_feat_emb.shape[0],
                                        scaling=self.context_clip_rope_scaling,
                                        device=device,
                                    )
                                clip_feat_emb = rope_apply_tokens_1d(
                                    clip_feat_emb.unsqueeze(0), clip_freqs
                                ).squeeze(0)
                        else:
                            clip_feat_emb = torch.zeros(self.clip_len, self.dim, device=device, dtype=dtype)
                    else:
                        clip_feat_emb = torch.zeros(self.clip_len, self.dim, device=device, dtype=dtype)
                    context_parts.append(clip_feat_emb)

                # Sync 特征：直接与 text、clip concat，不足补零，超过截断到 sync_len；无 Sync 时补零
                if self.sync_emb is not None:
                    if sync_features is not None and i < len(sync_features) and sync_features[i] is not None:
                        sync_feat = sync_features[i].to(device=device, dtype=dtype)
                        if sync_feat.ndim == 3:
                            sync_feat = sync_feat.squeeze(0)  # (T_sync, 768)
                        if sync_feat.ndim == 2 and sync_feat.shape[-1] == 768:
                            T = sync_feat.size(0)
                            if T < self.sync_len:
                                pad = torch.zeros(self.sync_len - T, 768, device=device, dtype=dtype)
                                sync_feat = torch.cat([sync_feat, pad], dim=0)
                            else:
                                sync_feat = sync_feat[:self.sync_len]
                            sync_feat_emb = self.sync_emb(sync_feat)  # (sync_len, dim)
                            if self.use_sync_context_rope and self.sync_context_freqs is not None:
                                sync_freqs = self._build_continuous_context_rope_freqs(
                                    sync_feat_emb.shape[0],
                                    target_audio_len,
                                    self.context_sync_rope_scaling,
                                    device,
                                    source_fps=self.sync_fps,
                                )
                                if sync_freqs is None:
                                    sync_freqs = self._get_context_rope_freqs(
                                        name="sync_context_freqs",
                                        length=sync_feat_emb.shape[0],
                                        scaling=self.context_sync_rope_scaling,
                                        device=device,
                                    )
                                sync_feat_emb = rope_apply_tokens_1d(
                                    sync_feat_emb.unsqueeze(0), sync_freqs
                                ).squeeze(0)
                        else:
                            sync_feat_emb = torch.zeros(self.sync_len, self.dim, device=device, dtype=dtype)
                    else:
                        sync_feat_emb = torch.zeros(self.sync_len, self.dim, device=device, dtype=dtype)
                    if not self.sync_drop_from_context:
                        context_parts.append(sync_feat_emb)

                combined_context = torch.cat(context_parts, dim=0)  # (max_context_len, dim)
                video_contexts.append(combined_context)

            context = torch.stack(video_contexts)  # [B, max_context_len, dim]
            context_lens = torch.full(
                (batch_size,),
                context.shape[1],
                device=context.device,
                dtype=torch.long,
            )
        else:
            context = text_context
            context_lens = None
            if clip_fea is not None:
                # 原有的 image-to-video 逻辑
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)

        # Scheme A: add aligned sync to audio tokens
        sync_aligned = None
        if self.is_audio_type and self.sync_add_to_audio and sync_features is not None:
            sync_aligned = self._align_sync_to_audio(
                sync_features=sync_features,
                seq_lens=seq_lens,
                seq_len=seq_len,
                device=x.device,
                dtype=x.dtype,
            )
            if sync_aligned is not None:
                if self.use_sp:
                    pad_size = (-seq_len) % self.sp_size
                    if pad_size > 0:
                        pad = torch.zeros(sync_aligned.shape[0], pad_size, sync_aligned.shape[2], device=x.device, dtype=x.dtype)
                        sync_aligned = torch.cat([sync_aligned, pad], dim=1)
                    sync_aligned = torch.chunk(sync_aligned, self.sp_size, dim=1)[self.sp_rank]
                if not self.sync_add_to_audio_each_block:
                    x = x + sync_aligned

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            sync_aligned=sync_aligned if (self.sync_add_to_audio_each_block or self.sync_use_adaln) else None)

        return x, e, kwargs
        
    def post_transformer_block_out(self, x, grid_sizes, e):
        # head
        x = self.head(x, e)
        if self.use_sp: 
            x = all_gather(x, dim=1)
        # unpatchify
        if self.is_audio_type:
            ## grid_sizes is [B 1] where 1 is L, 
            # converting grid_sizes from [B 1] -> [B]
            grid_sizes = [gs[0] for gs in grid_sizes]
            assert len(x) == len(grid_sizes)
            x = [u[:gs] for u, gs in zip(x, grid_sizes)]
        else:
            ## grid_sizes is [B 3] where 3 is F H w
            x = self.unpatchify(x, grid_sizes)

        return [u.bfloat16() for u in x]


    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        clip_features=None,
        sync_features=None,
        y=None,
        first_frame_is_clean=False
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
                OR 
                List of input audio tensors, each with shape [L, C_in]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            clip_features (List[Tensor], *optional*):
                List of CLIP video features, each with shape [T_clip, 1024]
            sync_features (List[Tensor], *optional*):
                List of Synchformer video features, each with shape [T_sync, 768]
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
                OR
                List of denoised audio tensors with original input shapes [L, C_in]
        """
        x, e, kwargs = self.prepare_transformer_block_kwargs(
            x=x,
            t=t,
            context=context,
            seq_len=seq_len,
            clip_fea=clip_fea,
            clip_features=clip_features,
            sync_features=sync_features,
            y=y,
            first_frame_is_clean=first_frame_is_clean
        )

        for block in self.blocks:
            x = gradient_checkpointing(
                    enabled=(self.training and self.gradient_checkpointing),
                    module=block,
                    x=x,
                    **kwargs
                )

        return self.post_transformer_block_out(x, kwargs['grid_sizes'], e)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            # v is [F H w] F * H * 80, 100, it was right padded by 20. 
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        # out is list of [C F H W]
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        if self.is_video_type:
            assert isinstance(self.patch_embedding, nn.Conv3d), f"Patch embedding for video should be a Conv3d layer, got {type(self.patch_embedding)}"
            nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
        for block in self.blocks:
            if block.sync_adaln is not None:
                nn.init.zeros_(block.sync_adaln[-1].weight)
                nn.init.zeros_(block.sync_adaln[-1].bias)
