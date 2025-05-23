# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional, Tuple, Union
from dataclasses import dataclass
import math
import functools

import torch
from torch import nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear
)
from ..peft import LoraColumnParallelLinear, LoraRowParallelLinear

from apex.normalization import FusedRMSNorm as RMSNorm
from transformers import Blip2Processor, Blip2Model

import configs.global_configs
if configs.global_configs.USE_FLASH_ATTENTION:
    from flash_attn import flash_attn_func

default_linear_init = functools.partial(nn.init.kaiming_uniform_, a=math.sqrt(5))

from .llama import precompute_freqs_cis, reshape_for_broadcast, apply_rotary_emb, repeat_kv


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000

    max_batch_size: int = 32
    max_seq_len: int = 2048

    rope_scaling: Optional[float] = None

    lora_rank: int = -1 # lora

    bias_tuning: bool = True  # bias


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = LoraColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=args.bias_tuning,
            gather_output=False,
            init_method=default_linear_init,
            lora_rank=args.lora_rank
        )
        self.wk = LoraColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=args.bias_tuning,
            gather_output=False,
            init_method=default_linear_init,
            lora_rank=args.lora_rank
        )
        self.wv = LoraColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=args.bias_tuning,
            gather_output=False,
            init_method=default_linear_init,
            lora_rank=args.lora_rank
        )
        self.wo = LoraRowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=args.bias_tuning,
            input_is_parallel=True,
            init_method=default_linear_init,
            lora_rank=args.lora_rank
        )

        self.args = args

        self.flash = configs.global_configs.USE_FLASH_ATTENTION
        self.k_cache, self.v_cache = None, None

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
        mask: Union[torch.Tensor, str, None]
    ) -> torch.Tensor:
        """
        Supported mask spec:
        1. Float tensor: The tensor is added to the attention score matrix.
        2. Boolean tensor: Substitute the ``True`` values with ``0.0`` and ``False`` values with
           ``-inf``, then process in the same way as the float tensor.
        3. str: Currently the only supported choice is ``causal``, for which each token attends
           to all tokens appearing no later than itself. Our implementation assumes the query and
           key sequences aligns on the right for ``causal`` if their lengths are not equal.
        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # if cache is enabled, prepend keys and values in the history.
        if self.k_cache is None or self.v_cache is None:
            keys, values = xk, xv
        else:
            self.k_cache = self.k_cache.to(xk)
            self.v_cache = self.v_cache.to(xv)
            self.k_cache[:bsz, start_pos: start_pos + seqlen, :, :] = xk
            self.v_cache[:bsz, start_pos: start_pos + seqlen, :, :] = xv
            keys = self.k_cache[:bsz, :start_pos + seqlen]
            values = self.v_cache[:bsz, :start_pos + seqlen]

        is_causal = isinstance(mask, str) and mask == "causal"
        # "causal" dispatches to flash_attn only when q and k have the same seqlen
        # because currently the flash_attn causal impl for unequal q & k length is not suited
        # for generation: Generation with cache requires aligning on the right, while the
        # current flash_attn impl aligns on the left. For example, we expect the mask to be
        # as the left one, while the current flash_attn impl gives the right one
        #
        #              K                     K
        #        1 1 1 1 1 0 0         1 0 0 0 0 0 0
        #     Q  1 1 1 1 1 1 0       Q 1 1 0 0 0 0 0
        #        1 1 1 1 1 1 1         1 1 1 0 0 0 0
        use_flash = (
            self.flash  # user configuration
            and (mask is None or (is_causal and keys.size(1) == xq.size(1)))  # supported mask
        )
        if use_flash:
            # repeating k/v heads is included in flash_attn
            output = flash_attn_func(xq, keys, values, dropout_p=0.0, causal=is_causal)
            output = output.contiguous().view(bsz, seqlen, -1)
        else:
            # repeat k/v heads if n_kv_heads < n_heads
            keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
            values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

            xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            if isinstance(mask, str):
                if is_causal:
                    mask = self._make_causal_mask(xq.size(2), keys.size(2))
                    mask = mask.to(xq.device, non_blocking=True)
                else:
                    raise NotImplementedError()
            output = F.scaled_dot_product_attention(xq, keys, values, dropout_p=0.0, attn_mask=mask)
            output = output.transpose(
                1, 2
            ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)

    def allocate_kv_cache(self, max_batch_size: int, max_seq_len: int) -> None:
        kv_cache_shape = (max_batch_size, max_seq_len, self.n_local_kv_heads, self.head_dim)
        if self.k_cache is None or self.k_cache.size() != kv_cache_shape:
            self.k_cache = torch.empty(kv_cache_shape)
        if self.v_cache is None or self.v_cache.size() != kv_cache_shape:
            self.v_cache = torch.empty(kv_cache_shape)

    def destroy_kv_cache(self) -> None:
        self.k_cache, self.v_cache = None, None

    def _make_causal_mask(self, q_len: int, kv_len: int) -> torch.Tensor:
        q_indices = torch.arange(q_len) - q_len
        kv_indices = torch.arange(kv_len) - kv_len
        causal_mask_bool = q_indices.view(-1, 1) >= kv_indices.view(1, -1)
        return causal_mask_bool

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        args: ModelArgs,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = LoraColumnParallelLinear(
            dim, hidden_dim, bias=args.bias_tuning, gather_output=False,
            init_method=default_linear_init, lora_rank=args.lora_rank
        )
        self.w2 = LoraRowParallelLinear(
            hidden_dim, dim, bias=args.bias_tuning, input_is_parallel=True,
            init_method=default_linear_init, lora_rank=args.lora_rank
        )
        self.w3 = LoraColumnParallelLinear(
            dim, hidden_dim, bias=args.bias_tuning, gather_output=False,
            init_method=default_linear_init, lora_rank=args.lora_rank
        )

    # @torch.compile
    def _silu_gating(self, x, y):
        return F.silu(x) * y

    def forward(self, x):
        return self.w2(self._silu_gating(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            args=args
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def _forward_ffn(self, h):
        return h + self.feed_forward(self.ffn_norm(h))

    def _forward_attention(self, x, start_pos, freqs_cis, mask):
        return x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
        mask: Union[torch.Tensor, str, None]
    ) -> torch.Tensor:
        h = self._forward_attention(x, start_pos, freqs_cis, mask)
        out = self._forward_ffn(h)
        return out


class Transformer(nn.Module):
    is_peft = True
    def __init__(self, params: ModelArgs, with_visual=False):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=default_linear_init
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=default_linear_init
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2,
            theta=self.params.rope_theta, scaling=self.params.rope_scaling
        )

        self.image_words = 0
        self.cache_image_words = 0 # for inference
        if with_visual:
            print("build llama model with qformerv2")
            self.qformer = Blip2Model.from_pretrained("/home/cx/ckpts/blip2-opt-2.7b", torch_dtype=torch.float16)  #("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

            self.qformer.language_projection = None
            self.qformer.language_model = None

            self.qformer_proj = nn.Sequential(
                nn.Linear(768, params.dim),
                nn.LayerNorm(params.dim)
            )
            self.image_words = 32
            # add image tags
            self.start_img = nn.Parameter(torch.rand(1, 1, params.dim))
            self.end_img = nn.Parameter(torch.rand(1, 1, params.dim))


    def get_trainable_params(self):
        trainable = {}
        for name, para in self.named_parameters():
            if not name.startswith("qformer."):
                trainable_key_words = ['norm', 'bias', 'lora']
                if any([_ in name for _ in trainable_key_words]):
                    trainable[name] = para

        return trainable


    def mc_dropout_and_mask(self, features, dropout_rate=0.5, mc_iterations=10, mask_percentage=0.25):
        # features should be of shape [B, L, D]
        # Apply MC Dropout
        mc_samples = [F.dropout(features, p=dropout_rate, training=True) for _ in range(mc_iterations)]
        mc_samples_stacked = torch.stack(mc_samples, dim=0)

        # Calculate variance across the MC samples
        variance = torch.var(mc_samples_stacked, dim=0)
        
        # Calculate mean variance across the feature dimension D
        mean_variance = torch.mean(variance, dim=-1)  # Shape: [B, L]

        # Flatten the mean variance to rank the elements
        flat_mean_variance = mean_variance.view(-1)

        # Determine the threshold based on the desired percentage
        k = int(flat_mean_variance.numel() * (1 - mask_percentage))
        # We use kthvalue to find the threshold that will give us the desired sparsity
        threshold = flat_mean_variance.kthvalue(k).values.item()

        # Create mask: if variance is less than the threshold, the feature is important
        mask = (mean_variance < threshold).float()
        mask = mask.unsqueeze(-1)
        return mask
    
    def encode_image(self, image):
        # [B, 32, 768]
        with torch.no_grad():
            image_feats = self.qformer.get_qformer_features(pixel_values=image)
        # image_feats = self.qformer_proj(image_feats.last_hidden_state)
        return image_feats

    def forward(self, examples, image=None):
        self._destroy_kv_cache()  # training always disables kv cache
        _bsz, seqlen = examples.shape
        h = self.tok_embeddings(examples)
        self.freqs_cis = self.freqs_cis.to(h.device)

        image_words = 0
        if image is not None:
            h_bos, h_caption = h[:, :1], h[:, 1:]
            image_tokens = self.encode_image(image)
            image_tokens = self.qformer_proj(image_tokens.last_hidden_state)
            h = torch.cat((h_bos, self.start_img.expand(_bsz, -1, -1), image_tokens, self.end_img.expand(_bsz, -1, -1), h_caption), dim=1)
            image_words = image_tokens.shape[1] + 1 + 1
            seqlen = h.shape[1]

        freqs_cis = self.freqs_cis[:seqlen]
        for layer in self.layers:
            h = layer(h, start_pos=0, freqs_cis=freqs_cis, mask="causal")
        h = self.norm(h)
        output = self.output(h[:, image_words:, :])
        return output


    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor, start_pos: int, image=None):
        _bsz, seqlen = tokens.shape
        if start_pos == 0:
            self._allocate_kv_cache(_bsz)  # kv cache will not re-allocate if size is unchanged
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        if image is not None:
            assert start_pos == 0
            h_bos, h_caption = h[:, :1], h[:, 1:]
            #-------UTS------------------------------------------------------
            image_tokens = self.encode_image(image).last_hidden_state
            mask = self.mc_dropout_and_mask(image_tokens)
            image_tokens = image_tokens * mask
            image_tokens = self.qformer_proj(image_tokens)
            #----------------------------------------------------------------
            self.cache_image_words = image_tokens.shape[1] + 1 + 1
            h = torch.cat((h_bos, self.start_img.repeat(_bsz, 1, 1), image_tokens, self.end_img.repeat(_bsz, 1, 1), h_caption), dim=1)
            seqlen = h.shape[1]
            freqs_cis = self.freqs_cis[0: seqlen]
        else:
            image_tokens = None
            if start_pos == 0:
                self.cache_image_words = 0
                freqs_cis = self.freqs_cis[0: seqlen]
            else:
                # if image was not None when start_pos=0,
                # the offset should be added to start_pos within later forward_inference calls
                start_pos = start_pos + self.cache_image_words
                freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        # Despite that "causal" also works for seqlen == 1, keep it to None for possibly
        # better performance
        mask = None if seqlen == 1 else "causal"

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float(), image_tokens

    def _allocate_kv_cache(self, max_batch_size: int) -> None:
        for layer in self.layers:
            layer.attention.allocate_kv_cache(max_batch_size, self.params.max_seq_len)

    def _destroy_kv_cache(self) -> None:
        for layer in self.layers:
            layer.attention.destroy_kv_cache()


