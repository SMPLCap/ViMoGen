import math

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .utils import hash_state_dict_keys
from copy import deepcopy

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings


__all__ = ['WanModel']


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        q = q.unsqueeze(0).transpose(1, 2).to(dtype)
        k = k.unsqueeze(0).transpose(1, 2).to(dtype)
        v = v.unsqueeze(0).transpose(1, 2).to(dtype)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).contiguous()

    # output
    return x.type(out_dtype)


def create_sdpa_mask(q, k, q_lens, k_lens, causal=False):
    b, lq, lk = q.size(0), q.size(1), k.size(1)
    if q_lens is None:
        q_lens = torch.tensor([lq] * b, dtype=torch.int32)
    if k_lens is None:
        k_lens = torch.tensor([lk] * b, dtype=torch.int32)
    attn_mask = torch.zeros((b, lq, lk), dtype=torch.bool)
    for i in range(b):
        q_len, k_len = q_lens[i], k_lens[i]
        attn_mask[i, q_len:, :] = True
        attn_mask[i, :, k_len:] = True
        
        if causal:
            causal_mask = torch.triu(torch.ones((lq, lk), dtype=torch.bool), diagonal=1)
            attn_mask[i, :, :] = torch.logical_or(attn_mask[i, :, :], causal_mask)

    attn_mask = attn_mask.logical_not().to(q.device, non_blocking=True)
    return attn_mask


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn('Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.')
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out



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


@amp.autocast(enabled=False, device_type="cuda")
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


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

    def forward(self, x, seq_lens, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        # import pdb; pdb.set_trace()
        q, k, v = qkv_fn(x)
        x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        """
        x:              [B, L1, C].
        context:        [B, L2, C].
        context_lens:   [B].
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


WANX_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
}

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
                ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WANX_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))
        
        # m2m cross attn
        # self.ref_motion_cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=0)
        self.ref_motion_cross_attn = WanT2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm4 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)


    def forward(
        self,
        x,        
        e,
        seq_lens,
        freqs,
        context,
        context_lens,
        ref_motion,    # [B, T, C]
        ref_motion_lens,   # [B]
        attend_to_text_mask=None,  # [B]: 1 for attend to text, 0 for attend to ref motion
    ):
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32, device_type="cuda"):
            e = (self.modulation.to(dtype=e.dtype, device=e.device) + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, 
            freqs)

        with amp.autocast(dtype=torch.float32, device_type="cuda"):
            x = x + y * e[2]

        # Cross-attention based on attend_to_text_mask
        text_mask = attend_to_text_mask == 1
        ref_motion_mask = ~text_mask

        # Dummy way to get output shape
        b, s, d = ref_motion.shape
        # combined_out = self.cross_attn(self.norm3(x), context, None)
        combined_out = torch.zeros([b, s, self.dim], dtype=context.dtype, device=context.device)

        # Process text and reference motion separately
        attend_to_text_mask = attend_to_text_mask.to(dtype=bool)
        have_text = attend_to_text_mask.sum() > 0
        have_ref_motion = (~attend_to_text_mask).sum() > 0 
        if have_text:
            text_out = self.cross_attn(self.norm3(x[text_mask]), context[text_mask], None)
            # print('text_out shape', text_out.shape)
            combined_out[text_mask] = text_out

        if have_ref_motion:
            ref_motion_out = self.ref_motion_cross_attn(self.norm4(x[ref_motion_mask]), ref_motion[ref_motion_mask], ref_motion_lens[ref_motion_mask])
            # print('ref_motion_out shape', ref_motion_out.shape)
            combined_out[ref_motion_mask] = ref_motion_out if x[ref_motion_mask].size(0) > 0 else 0
        
        x += combined_out

        # ffn function
        y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
        with amp.autocast(dtype=torch.float32, device_type="cuda"):
            x = x + y * e[5]
        return x


class WanAttentionBlock_SAonly(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)


    def forward(
        self,
        x,        
        e,
        seq_lens,
        freqs,
    ):
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32, device_type="cuda"):
            e = (self.modulation.to(dtype=e.dtype, device=e.device) + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, 
            freqs)

        with amp.autocast(dtype=torch.float32, device_type="cuda"):
            x = x + y * e[2]

        # ffn function
        y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
        with amp.autocast(dtype=torch.float32, device_type="cuda"):
            x = x + y * e[5]
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
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32, device_type="cuda"):
            e = (self.modulation.to(dtype=e.dtype, device=e.device) + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, C]
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0), :]
        x = x.permute(1, 0, 2)
        return x

class WanModelTM2M(nn.Module):

    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6, 
                 ref_motion_dim=16,
                 **kwargs):
        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

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

        # embeddings
        self.motion_embedding = nn.Linear(in_dim, dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.motion_pos_embedding = PositionalEncoding(d_model=dim, max_len=5000)

        # ref motion
        self.ref_motion_embedding = nn.Linear(ref_motion_dim, dim)
        # self.ref_motion_embedding = nn.Sequential(
        #     nn.Linear(ref_motion_dim, dim), 
        #     nn.SiLU(), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim), 
        #     nn.SiLU(), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.ref_motion_pos_embedding = PositionalEncoding(d_model=dim, max_len=5000)
        self.ref_motion_encoder = nn.ModuleList([
            WanAttentionBlock_SAonly(dim, ffn_dim, num_heads, window_size, qk_norm, eps)
            for _ in range(num_layers//2)
        ])

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                               window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        patch_size = (1,)
        self.motion_head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

    def get_fsdp_wrap_module_list(self, ):
        return list(self.blocks)
    
    def forward(
        self,
        x,
        timestep,
        context,
        clip_fea=None,
        y=None,
        use_gradient_checkpointing=False,
        x_mask=None,
        ref_motion=None,
        ref_motion_mask=None,
        attend_to_text_mask=None,
    ):
        """
        x:              motion latents of shape [B, T, C].
        x_mask:         mask of shape [B, T], 1 for valid, 0 for invalid.
        t:              [B].
        context:        text features of shape [B, L, C].
        ref_motion:     [B, T, C].
        ref_motion_mask: [B, T], 1 for valid, 0 for invalid.
        attend_to_text_mask: [B], 1 for attend to text, 0 for attend to ref motion.
        """
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        if attend_to_text_mask is None:
            # if ref motion is not available, it will be set as all zeros tensor.
            attend_to_text_mask = ((ref_motion**2).sum(dim=[1,2]) == 0)
            # print('computed attend_to_text_mask:', attend_to_text_mask)

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = x[0].device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # embeddings
        x = self.motion_embedding(x)
        x = self.motion_pos_embedding(x)

        # ref motion
        if ref_motion is not None:
            ref_motion = self.ref_motion_embedding(ref_motion)
            ref_motion = self.ref_motion_pos_embedding(ref_motion)

        # get seq_lens based on x_mask
        if x_mask is None:
            x_mask = torch.ones((x.size(0), x.size(1)), device=x.device)
        seq_lens = x_mask.sum(1).to(torch.long)

        # time embeddings
        with amp.autocast(dtype=torch.float32, device_type="cuda"):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, timestep).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(context)

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if ref_motion is not None:
            # arguments
            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                freqs=self.freqs,
            )
            for block in self.ref_motion_encoder:
                if self.training and use_gradient_checkpointing:
                    ref_motion = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        ref_motion, **kwargs,
                        use_reentrant=False,
                    )
                else:
                    ref_motion = block(ref_motion, **kwargs)

        ref_motion_lens = ref_motion_mask.sum(1).to(torch.long)
        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            ref_motion=ref_motion,
            ref_motion_lens=deepcopy(ref_motion_lens),
            attend_to_text_mask=attend_to_text_mask,
        )
                
        # import pdb; pdb.set_trace()
        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # head
        x = self.motion_head(x, e)

        return x

    def init_weights(self):
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        # nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        # nn.init.zeros_(self.motion_head.head.weight)

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    
    
class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        return state_dict
    
    def from_civitai(self, state_dict):
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "model_type": "i2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict, config
