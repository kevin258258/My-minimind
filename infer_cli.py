import argparse
import os
import re

import torch
from transformers import AutoTokenizer

from model.model import HollowStoneMindConfig, HollowStoneMindForCausalLM
from model.model_lora import apply_lora, load_lora, merge_lora


def resolve_default_model_path():
    """自动探测可用的模型权重，优先顺序：sft > dpo > pretrain"""
    candidates = [
        os.path.join("out", "sft_512.pth"),
        os.path.join("out", "sft_512_moe.pth"),
        os.path.join("out", "dpo_512.pth"),
        os.path.join("out", "dpo_512_moe.pth"),
        os.path.join("out", "pretrain_512.pth"),
        os.path.join("out", "pretrain_512_moe.pth"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="HollowStoneMind terminal chat inference tool"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=resolve_default_model_path(),
        help="Path to the exported model weight file",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="model",
        help="Path to tokenizer directory",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=512,
        help="Model hidden size used during training",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=8,
        help="Number of transformer blocks used during training",
    )
    parser.add_argument(
        "--use_moe",
        type=int,
        default=-1,
        choices=[-1, 0, 1],
        help="-1 auto detect from weight, 0 disable MoE, 1 enable MoE",
    )
    parser.add_argument(
        "--num_experts_per_tok",
        type=int,
        default=2,
        help="Top-k experts per token (for MoE)",
    )
    parser.add_argument(
        "--n_routed_experts",
        type=int,
        default=0,
        help="Number of routed experts (0 means infer from checkpoint when possible)",
    )
    parser.add_argument(
        "--n_shared_experts",
        type=int,
        default=0,
        help="Number of shared experts (0 means infer from checkpoint when possible)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype on device",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum generated tokens per turn",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Penalty for repeated tokens",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="",
        help="Optional system prompt",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Run a single prompt and exit instead of interactive chat",
    )
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="auto",
        choices=["auto", "raw", "qa", "chat_template"],
        help="Prompt formatting strategy for generation",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA weights (optional). If provided, LoRA will be merged for inference.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank (must match training config)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha (must match training config)",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="LoRA target modules, comma separated",
    )
    return parser.parse_args()


def resolve_dtype(device: str, dtype_name: str):
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if device.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def build_model(args):
    raw_state = torch.load(args.model_path, map_location="cpu")
    if "model" in raw_state and isinstance(raw_state["model"], dict):
        raw_state = raw_state["model"]

    normalized_state_dict = {}
    for key, value in raw_state.items():
        normalized_key = (
            key.replace(".self_attn.", ".attention.")
            .replace(".mlp.experts.", ".mlp.routed_experts.")
        )
        normalized_state_dict[normalized_key] = value

    has_moe_weights = any(
        ".mlp.routed_experts." in key for key in normalized_state_dict.keys()
    )

    def infer_expert_count(prefix: str):
        ids = set()
        pattern = re.compile(rf"{re.escape(prefix)}(\d+)\.")
        for k in normalized_state_dict.keys():
            m = pattern.search(k)
            if m:
                ids.add(int(m.group(1)))
        return (max(ids) + 1) if ids else 0

    inferred_routed = infer_expert_count(".mlp.routed_experts.")
    if inferred_routed == 0:
        for k, v in normalized_state_dict.items():
            if k.endswith(".mlp.gate.weight") and hasattr(v, "shape") and len(v.shape) == 2:
                inferred_routed = int(v.shape[0])
                break
    inferred_shared = infer_expert_count(".mlp.shared_experts.")

    if args.use_moe == -1:
        use_moe = has_moe_weights
    else:
        use_moe = bool(args.use_moe)

    n_routed = args.n_routed_experts if args.n_routed_experts > 0 else inferred_routed
    if args.n_shared_experts > 0:
        n_shared = args.n_shared_experts
    elif inferred_shared > 0:
        n_shared = inferred_shared
    else:
        n_shared = 1 if use_moe else 0

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    config = HollowStoneMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=use_moe,
        num_experts_per_tok=args.num_experts_per_tok,
        n_routed_experts=n_routed if n_routed > 0 else (4 if use_moe else 0),
        n_shared_experts=n_shared,
    )
    model = HollowStoneMindForCausalLM(config)

    missing, unexpected = model.load_state_dict(normalized_state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # LoRA support
    if getattr(args, "lora_path", None):
        adapted = apply_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
        )
        if not adapted:
            raise ValueError(f"No modules matched for LoRA injection with target_modules={args.lora_target_modules!r}")
        load_lora(model, args.lora_path)
        print(f"[INFO] LoRA loaded from {args.lora_path}, modules: {', '.join(adapted)}")
        # merge for faster inference
        model = merge_lora(model)
        print("[INFO] LoRA merged into base model for inference.")

    dtype = resolve_dtype(args.device, args.dtype)
    model = model.to(args.device)
    if args.device.startswith("cuda"):
        model = model.to(dtype=dtype)
    model.eval()

    return tokenizer, model

def format_prompt_raw(messages):
    parts = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            parts.append(content)
        elif role == "user":
            parts.append(content)
        elif role == "assistant":
            parts.append(content)
    return "\n".join(part for part in parts if part)


def format_prompt_qa(messages):
    parts = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            parts.append(f"系统提示：{content}")
        elif role == "user":
            parts.append(f"问题：{content}")
        elif role == "assistant":
            parts.append(f"回答：{content}")
    parts.append("回答：")
    return "\n".join(parts)


def resolve_prompt_format(tokenizer, args):
    if args.prompt_format != "auto":
        return args.prompt_format

    model_name = os.path.basename(args.model_path).lower()
    if any(tag in model_name for tag in ("sft", "lora", "dpo")):
        return "chat_template"

    if getattr(tokenizer, "chat_template", None):
        return "chat_template"

    return "qa"


def format_prompt(tokenizer, messages, prompt_format):
    if prompt_format == "chat_template":
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if prompt_format == "raw":
        return format_prompt_raw(messages)
    return format_prompt_qa(messages)


def generate_reply(tokenizer, model, messages, args):
    prompt_format = resolve_prompt_format(tokenizer, args)
    prompt_text = format_prompt(tokenizer, messages, prompt_format)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(args.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id,
            use_cache=True,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return reply or "..."


def run_single_prompt(tokenizer, model, args):
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    print(generate_reply(tokenizer, model, messages, args))


def run_interactive_chat(tokenizer, model, args):
    print("HollowStoneMind CLI")
    print(f"prompt_format={resolve_prompt_format(tokenizer, args)}")
    print("输入内容开始对话，输入 /clear 清空上下文，输入 /exit 退出。")

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            print("Bye.")
            break
        if user_input in {"/clear", "/reset"}:
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            print("Context cleared.")
            continue

        messages.append({"role": "user", "content": user_input})
        reply = generate_reply(tokenizer, model, messages, args)
        messages.append({"role": "assistant", "content": reply})
        print(f"HollowStoneMind> {reply}")


def main():
    args = parse_args()
    tokenizer, model = build_model(args)

    if args.prompt is not None:
        run_single_prompt(tokenizer, model, args)
        return

    run_interactive_chat(tokenizer, model, args)


if __name__ == "__main__":
    main()
