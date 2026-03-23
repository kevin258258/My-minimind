from typing import Optional, Tuple

from transformers import PretrainedConfig


class HollowStoneMindConfig(PretrainedConfig):
    model_type = "hollowstonemind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


MokioMindConfig = HollowStoneMindConfig

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast


# 现在我们先来编写数据整理的RMSnorm，当然，作为layer我们要继承一个module类
# 另外这里统一一下，向量统一形状都是B H S D，B是batch_size，H是head数量，S是序列长度，D是每个head的维度
class RMSNorm(nn.Module):
    # 这里我们需要编写初始化，forward等方法
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    # norm
    def _norm(self, x):
        x_float = x.float()
        return x_float * torch.rsqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )

    # forward
    def forward(self, x):
        return self.weight * self._norm(x).type_as(x)


def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {dim}")

    half_dim = dim // 2
    i = torch.arange(0, dim, step=2, dtype=torch.float32)
    freqs = 1.0 / (rope_base ** (i / dim))
    att = 1.0

    if rope_scaling is not None:
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )
        if end > orig_max:

            def inv(b):
                return (dim * math.log(orig_max / (2 * math.pi * b))) / (
                    2 * math.log(rope_base)
                )

            # 现在来计算高低频对应的index
            low = max(0, math.floor(inv(beta_slow)))
            high = min(half_dim - 1, math.ceil(inv(beta_fast)))
            ramp = torch.arange(half_dim, dtype=freqs.dtype, device=freqs.device)
            ramp = torch.clamp((ramp - low) / max(high - low, 0.001), 0, 1)

            freqs = freqs * (1 - ramp + ramp / factor)
            att = attn_factor
    positions = torch.arange(0, end, dtype=freqs.dtype, device=freqs.device)
    table = torch.outer(positions, freqs)
    cos = torch.cos(table).repeat_interleave(2, dim=-1) * att
    sin = torch.sin(table).repeat_interleave(2, dim=-1) * att

    return cos, sin


def apply_rope(cos, sin, Q, K, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        if d % 2 != 0:
            raise ValueError(f"Last dimension must be even, got {d}")

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]
    elif cos.dim() == 2:
        seq_len = Q.shape[-2]
        cos = cos[:seq_len].unsqueeze(0)
        sin = sin[:seq_len].unsqueeze(0)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = Q * cos + rotate_half(Q) * sin
    k_embed = K * cos + rotate_half(K) * sin

    return q_embed, k_embed


# repeat_kv 的作用是将 K 和 V 在 heads 维度上进行“广播”或“物理复制”，使其维度变成 [Batch, 32_heads, Seq_len, Head_dim]。
def repeat_kv(x: torch.Tensor, rep: int):
    if rep == 1:
        return x
    B, H, S, D = x.shape
    x = x.unsqueeze(2).expand(B, H, rep, S, D).reshape(B, H * rep, S, D)
    return x


class attention(nn.Module):
    def __init__(self, args: HollowStoneMindConfig):
        super().__init__()
        self.num_key_value_heads = (
            args.num_key_value_heads
            if args.num_key_value_heads is not None
            else args.num_attention_heads
        )
        assert args.num_attention_heads % self.num_key_value_heads == 0, (
            "num_attention_heads must be divisible by num_key_value_heads"
        )
        self.n_local_heads = args.num_attention_heads
        self.num_key_value_groups = self.n_local_heads // self.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(
            in_features=args.hidden_size,
            out_features=self.n_local_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            in_features=args.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            in_features=args.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            in_features=self.n_local_heads * self.head_dim,
            out_features=args.hidden_size,
            bias=False,
        )

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash_attention = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )

        # forward之前梳理一下逻辑
        # 我们先对输入的tensor（形状是[B,S,D])，进行投影能，得到B H S D
        # 然后我们需要对Q K V进行RoPE位置编码，得到新的Q K V
        # 然后KV记得先repat(对了注意kvcache)
        # 最后我们进行注意力计算，得到输出，最后经过线性层投影回[B,S,D]的形状，算出对应的加权value输出

    def forward(
        self,
        x: torch.Tensor,
        position_embeding: Tuple[torch.Tensor, torch.Tensor],
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        B, S, D = x.shape
        past_len = 0 if past_kv is None else past_kv[0].shape[2]
        xq = (
            self.q_proj(x).view(B, S, self.n_local_heads, self.head_dim).transpose(1, 2)
        )
        xk = (
            self.k_proj(x)
            .view(B, S, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        xv = (
            self.v_proj(x)
            .view(B, S, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        cos, sin = position_embeding
        position_ids = torch.arange(
            past_len, past_len + S, dtype=torch.long, device=x.device
        ).unsqueeze(0)
        xq, xk = apply_rope(
            cos, sin, xq, xk, position_ids=position_ids, unsqueeze_dim=1
        )

        if past_kv is not None:
            xk = torch.cat([past_kv[0], xk], dim=2)
            xv = torch.cat([past_kv[1], xv], dim=2)
        past_kv = (xk, xv) if use_cache else None

        kv_seq_len = xk.shape[-2]
        xk = repeat_kv(xk, self.num_key_value_groups)
        xv = repeat_kv(xv, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.shape[-1] < kv_seq_len:
                pad_len = kv_seq_len - attention_mask.shape[-1]
                attention_mask = F.pad(attention_mask, (pad_len, 0), value=1)
            elif attention_mask.shape[-1] > kv_seq_len:
                attention_mask = attention_mask[:, -kv_seq_len:]

        if self.flash_attention and S > 1 and kv_seq_len == S:
            attn_mask = None
            if attention_mask is not None and not torch.all(attention_mask == 1):
                attn_mask = (
                    attention_mask[:, None, None, :]
                    .expand(B, self.n_local_heads, S, kv_seq_len)
                    .bool()
                )
            out_put = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            past_len = kv_seq_len - S
            causal_mask = torch.triu(
                torch.full(
                    (S, kv_seq_len),
                    torch.finfo(scores.dtype).min,
                    device=scores.device,
                    dtype=scores.dtype,
                ),
                diagonal=1 + past_len,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

            if attention_mask is not None:
                extended_attention_mask = attention_mask[:, None, None, :].to(
                    dtype=scores.dtype
                )
                extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
                    scores.dtype
                ).min
                scores = scores + extended_attention_mask

            attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq)
            attn_weights = self.attn_dropout(attn_weights)
            out_put = attn_weights @ xv

        out_put = (
            out_put.transpose(1, 2)
            .contiguous()
            .view(B, S, self.n_local_heads * self.head_dim)
        )
        out_put = self.o_proj(out_put)
        out_put = self.resid_dropout(out_put)
        return out_put, past_kv


class FeedForward(nn.Module):
    # 这里主要有什么呢？
    # init
    # MLP，门控，dropout
    def __init__(self, args: HollowStoneMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8 / 3)
            args.intermediate_size = 64 * ((intermediate_size + 63) // 64)

        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]

    def forward(self, x: torch.Tensor):
        return self.dropout(
            self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        )


class MindBlock(nn.Module):
    def __init__(self, config: HollowStoneMindConfig, layer_id: int):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.attention = attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states,
        position_embeding,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states, present_kv = self.attention(
            self.input_layernorm(hidden_states),
            position_embeding,
            past_kv=past_kv,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_kv


class HollowStoneMindModel(nn.Module):
    def __init__(self, config: HollowStoneMindConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [MindBlock(config, i) for i in range(config.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 这里先把RoPE的计算先做好，放在模型里，避免每次计算都要重新计算RoPE
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids cannot be None")

        batch_size, seq_len = input_ids.shape

        if hasattr(past_kv, "layers"):
            past_kv = None

        past_kv = past_kv or [None] * len(self.layers)

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeding = (self.freqs_cos, self.freqs_sin)

        presents_kv = []
        for layer, layer_past_kv in zip(self.layers, past_kv):
            hidden_states, present_kv = layer(
                hidden_states,
                position_embeding,
                past_kv=layer_past_kv,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents_kv.append(present_kv)

        hidden_states = self.norm(hidden_states)

        return hidden_states, presents_kv


class HollowStoneMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = HollowStoneMindConfig
    base_model_prefix = "model"

    def __init__(self, config: HollowStoneMindConfig):
        super().__init__(config)
        self.model = HollowStoneMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[
            List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
        ] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
        ] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ):
        use_cache = (
            use_cache
            if use_cache is not None
            else getattr(self.config, "use_cache", True)
        )

        hidden_states, presents_kv = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_kv=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        logits = self.lm_head(hidden_states).float()
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits, presents_kv)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents_kv,
        )
