from transformers import PretrainedConfig
from typing import Optional


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

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

import torch
import math
import torch.nn as nn
from torch.nn import init
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
# 现在我们先来编写数据整理的RMSnorm，当然，作为layer我们要继承一个module类
class RMSNorm(nn.Module):
        #这里我们需要编写初始化，forward等方法
        def __init__(self,dim:int , eps:float = 1e-5):
             super().__init__()
             self.dim =dim
             self.eps =eps
             self.weight = nn.Parameter(torch.ones(dim))

        # norm
        def _norm(self,x):
             x_float = x.float()
             return x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # forward
        def forward(self,x):
             return self.weight * self._norm(x).type_as(x)

def precompute_freqs_cis (
          dim:int,
          end:int = int(32 * 1024),
          rope_base:float = 1e6,
          rope_scaling :Optional[dict] = None,):
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
                return (dim * math.log(orig_max / (2 * math.pi * b))) / (2 * math.log(rope_base))

             #现在来计算高低频对应的index
            low  = max(0,math.floor(inv(beta_slow)))
            high = min(half_dim - 1,math.ceil(inv(beta_fast)))
            ramp = torch.arange(half_dim, dtype=freqs.dtype, device=freqs.device)
            ramp = torch.clamp((ramp - low) / max(high - low, 0.001), 0, 1)
             
            freqs = freqs * (1 - ramp + ramp / factor)
            att = attn_factor
    positions = torch.arange(0, end, dtype=freqs.dtype, device=freqs.device)
    table = torch.outer(positions, freqs)
    cos = torch.cos(table).repeat_interleave(2,dim = -1) * att
    sin = torch.sin(table).repeat_interleave(2,dim = -1) * att

    return cos,sin

def  apply_rope(cos,sin,Q,K,position_ids = None,unsqueeze_dim = 1):
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

    return q_embed,k_embed
