# HollowStoneMind

从零复现 [MiniMind](https://github.com/jingyaogong/minimind) 的轻量级 LLM 训练项目，覆盖完整的预训练 → 后训练链路。

## 技术栈

### 模型架构

| 技术 | 说明 |
|------|------|
| **RMSNorm** | Root Mean Square Layer Normalization，替代传统 LayerNorm |
| **RoPE** | Rotary Position Embeddings，旋转式位置编码 |
| **YaRN** | Yet another RoPE extensioN，支持推理时位置外推（`original_max_position_embeddings`、`factor`、`beta_fast`/`beta_slow` 控制高低频插值） |
| **GQA** | Grouped Query Attention，通过 `num_key_value_heads` 控制 KV head 共享数（默认 2） |
| **Flash Attention** | 基于 `torch.nn.functional.scaled_dot_product_attention`（训练/推理自动切换 flash/causal 路径） |
| **SwiGLU FFN** | Gate + Up + Down 三投影前馈网络，SiLU 激活 |
| **MoE** | Mixture of Experts，可选启用。支持 top-k 门控路由（`num_experts_per_tok`）、routed experts + shared experts、load balancing aux loss |

### 预训练 (Pretraining)

- 自回归 causal language modeling，`<|im_start|>` / `<|im_end|>` 对话格式预处理
- 分词后自动包裹 BOS / EOS token，padding 至固定长度
- `config.max_position_embeddings = 32768`，理论支持长上下文

### 后训练 (Post-training)

| 方法 | 文件 | 说明 |
|------|------|------|
| **SFT** | `trainer/train_sft.py` | 监督微调，chat template 格式，assistant-only label masking，动态 system prompt 增强 |
| **LoRA** | `trainer/train_lora.py`, `model/model_lora.py` | 低秩适配微调，支持可配置 `lora_rank`/`lora_alpha`/`target_modules`，merge 回 base model 推理 |
| **DPO** | `trainer/train_dpo.py` | Direct Preference Optimization，chosen/rejected 对比学习，logsigmoid loss，reference model freeze |

### 训练基础设施

- **混合精度**：bfloat16 / float16 + `GradScaler`
- **梯度累积**：`accumulation_steps` 支持大有效 batch size
- **分布式训练**：DDP + `DistributedSampler`
- **余弦学习率调度**：`lr * (0.1 + 0.45 * (1 + cos(π * step / total)))`
- **Checkpoint**：完整 resume（model + optimizer + scaler + epoch + step），`SkipBatchSampler` 精确续训
- **日志**：终端 + WandB（可开关）

### 数据集处理

| 数据集 | 格式 | 用途 |
|--------|------|------|
| `PretrainDataset` | `{"text": "..."}` 文本续写 | 预训练 |
| `SFTDataset` | `{"conversations": [...]}` 多轮对话 | SFT / LoRA |
| `DPODataset` | `{"chosen": [...], "rejected": [...]}` | DPO |

SFT 数据处理关键设计：
- 通过增量 tokenize（逐条拼接 conversation 后对比长度）定位 assistant 回复区间，**避免硬编码 chat_template special token 格式**
- 随机 system prompt 增强（20% 概率），提升模型对 system prompt 的泛化
- 空 `<think>` 标签随机清理，适配 thinking 格式数据

## 项目结构

```
├── model/
│   ├── model.py          # 模型定义（Config / RMSNorm / RoPE+YaRN / Attention / MoE / FeedForward / CausalLM）
│   └── model_lora.py     # LoRA 模块（LoRALinear / apply / save / load / merge）
├── trainer/
│   ├── pretrain.py       # 预训练脚本
│   ├── train_sft.py      # SFT 训练脚本
│   ├── train_lora.py     # LoRA 训练脚本
│   ├── train_dpo.py      # DPO 训练脚本
│   └── trainer_utils.py  # 训练工具（LR 调度 / DDP 初始化 / Checkpoint 保存加载 / init_model）
├── dataset/
│   └── Im_dataset.py     # 数据集类（PretrainDataset / SFTDataset / DPODataset）
├── infer_cli.py          # 终端推理交互（支持 SFT / LoRA merge / DPO 权重）
├── tests/                # 单元测试
└── method/               # 算法组件原型（GQA / RMSNorm / RoPE）
```

## 快速开始

### 预训练

```bash
python trainer/pretrain.py \
    --data_path dataset/pretrain_hq.jsonl \
    --epochs 1 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --max_seq_len 512
```

### SFT 微调

```bash
python trainer/train_sft.py \
    --from_weight pretrain \
    --data_path dataset/sft_512.jsonl \
    --epochs 3 \
    --learning_rate 5e-4 \
    --max_seq_len 512
```

续训：

```bash
python trainer/train_sft.py --from_resume 1  # 其他参数同上
```

### LoRA 微调

```bash
python trainer/train_lora.py \
    --from_weight sft \
    --data_path dataset/lora_medical.jsonl \
    --lora_name lora_medical \
    --lora_rank 16 \
    --lora_alpha 32 \
    --learning_rate 5e-4
```

### DPO 对齐

```bash
python trainer/train_dpo.py \
    --from_weight sft \
    --data_path dataset/dpo.jsonl \
    --beta 0.1 \
    --learning_rate 5e-6
```

### 推理

```bash
# 交互对话
python infer_cli.py

# 加载 LoRA 权重推理（自动 merge）
python infer_cli.py --lora_path out/lora_medical_512.pth --lora_rank 8 --lora_alpha 16

# 单次 prompt
python infer_cli.py --prompt "你好，请介绍一下自己" --max_new_tokens 256
```

## License

MIT
