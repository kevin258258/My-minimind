import argparse
import os
import sys
import time
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from dataset.Im_dataset import DPODataset
from model.model import HollowStoneMindConfig, HollowStoneMindForCausalLM
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    init_model,
    is_main_process,
    lm_checkpoint,
    normalize_state_dict_keys,
    safe_torch_load,
    setup_seed,
)

warnings.filterwarnings("ignore")


def build_parser():
    parser = argparse.ArgumentParser(description="HollowStoneMind DPO")

    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "out"),
        help="模型保存目录",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints"),
        help="训练断点和完整checkpoint保存目录",
    )
    parser.add_argument("--save_weight", default="dpo", type=str, help="保存权重前缀")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO温度系数beta")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")

    parser.add_argument(
        "--accumulation_steps", type=int, default=8, help="梯度累积步数"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")

    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument("--max_seq_len", default=512, type=int, help="最大序列长度")
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "dataset", "dpo.jsonl"),
        help="DPO数据路径",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "model"),
        help="tokenizer路径",
    )
    parser.add_argument(
        "--from_weight",
        default="sft",
        type=str,
        help="policy初始化权重",
    )
    parser.add_argument(
        "--reference_weight",
        default="sft",
        type=str,
        help="reference模型权重（默认和from_weight一致）",
    )
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测并续训",
    )

    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument(
        "--wandb_project", type=str, default="HollowStoneMind-DPO", help="wandb项目名"
    )
    return parser


def sequence_log_probs(logits, labels):
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid_mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~valid_mask, 0)

    token_log_probs = torch.gather(
        F.log_softmax(shift_logits.float(), dim=-1), dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return (token_log_probs * valid_mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta,
):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    losses = -F.logsigmoid(logits)
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    reward_acc = (chosen_rewards > rejected_rewards).float().mean()
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean(), reward_acc


def train_epoch(epoch, loader, steps_per_epoch, start_step=0, wandb=None):
    start_time = time.time()

    for step, batch in enumerate(loader, start=start_step + 1):
        chosen_input_ids = batch["chosen_input_ids"].to(args.device)
        chosen_labels = batch["chosen_labels"].to(args.device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(args.device)
        rejected_input_ids = batch["rejected_input_ids"].to(args.device)
        rejected_labels = batch["rejected_labels"].to(args.device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(args.device)

        lr = get_lr(
            epoch * steps_per_epoch + step,
            args.epochs * steps_per_epoch,
            args.learning_rate,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            policy_chosen = model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask,
                use_cache=False,
            )
            policy_rejected = model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask,
                use_cache=False,
            )
            aux_chosen = getattr(policy_chosen, "aux_loss", None)
            aux_rejected = getattr(policy_rejected, "aux_loss", None)

            policy_chosen_logps = sequence_log_probs(policy_chosen.logits, chosen_labels)
            policy_rejected_logps = sequence_log_probs(
                policy_rejected.logits, rejected_labels
            )

            with torch.no_grad():
                ref_chosen = ref_model(
                    input_ids=chosen_input_ids,
                    attention_mask=chosen_attention_mask,
                    use_cache=False,
                )
                ref_rejected = ref_model(
                    input_ids=rejected_input_ids,
                    attention_mask=rejected_attention_mask,
                    use_cache=False,
                )
                ref_chosen_logps = sequence_log_probs(ref_chosen.logits, chosen_labels)
                ref_rejected_logps = sequence_log_probs(
                    ref_rejected.logits, rejected_labels
                )

            dpo_obj, chosen_reward, rejected_reward, reward_acc = dpo_loss(
                policy_chosen_logps=policy_chosen_logps,
                policy_rejected_logps=policy_rejected_logps,
                ref_chosen_logps=ref_chosen_logps,
                ref_rejected_logps=ref_rejected_logps,
                beta=args.beta,
            )

            aux_loss = None
            if aux_chosen is not None and aux_rejected is not None:
                aux_loss = 0.5 * (aux_chosen + aux_rejected)
            elif aux_chosen is not None:
                aux_loss = aux_chosen
            elif aux_rejected is not None:
                aux_loss = aux_rejected

            loss = dpo_obj if aux_loss is None else (dpo_obj + aux_loss)
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or step == steps_per_epoch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == steps_per_epoch:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / max(step - start_step, 1) * (
                steps_per_epoch - step
            ) // 60
            Logger(
                f"DPO Epoch:[{epoch + 1}/{args.epochs}]({step}/{steps_per_epoch}) "
                f"loss:{current_loss:.6f} lr:{current_lr:.12f} "
                f"chosen_reward:{chosen_reward.item():.6f} "
                f"rejected_reward:{rejected_reward.item():.6f} "
                f"reward_acc:{reward_acc.item():.4f} "
                f"epoch_Time:{eta_min}min:"
            )
            if wandb:
                wandb.log(
                    {
                        "loss": current_loss,
                        "lr": current_lr,
                        "chosen_reward": chosen_reward.item(),
                        "rejected_reward": rejected_reward.item(),
                        "reward_acc": reward_acc.item(),
                        "epoch_Time": eta_min,
                    }
                )

        if (step % args.save_interval == 0 or step == steps_per_epoch) and is_main_process():
            model.eval()
            moe_suffix = (
                "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
            )
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save({k: v.half() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir=args.checkpoint_dir,
            )
            model.train()


def build_reference_model(config, reference_weight, save_dir, device):
    ref = HollowStoneMindForCausalLM(config).to(device)
    if reference_weight != "none":
        moe_suffix = "_moe" if getattr(config, "use_moe", False) else ""
        weight_path = f"{save_dir}/{reference_weight}_{config.hidden_size}{moe_suffix}.pth"
        weights = safe_torch_load(weight_path, map_location=device, weights_only=True)
        weights = normalize_state_dict_keys(weights)
        ref.load_state_dict(weights, strict=False)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    return ref


def main():
    global args, lm_config, model, ref_model, optimizer, scaler, autocast_ctx

    parser = build_parser()
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"

    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = HollowStoneMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    ckp_data = (
        lm_checkpoint(
            lm_config,
            weight=args.save_weight,
            save_dir=args.checkpoint_dir,
        )
        if args.from_resume == 1
        else None
    )

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )

    wandb = None
    if args.use_wandb and is_main_process():
        import wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = (
            f"HollowStoneMind-DPO-Epoch-{args.epochs}-"
            f"BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        )
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            id=wandb_id,
            resume=resume,
        )

    model, tokenizer = init_model(
        lm_config,
        args.from_weight,
        tokenizer_path=args.tokenizer_path,
        save_dir=args.save_dir,
        device=args.device,
    )
    ref_model = build_reference_model(
        lm_config,
        args.reference_weight,
        args.save_dir,
        args.device,
    )

    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    local_samples = len(train_sampler) if train_sampler is not None else len(train_ds)
    steps_per_epoch = (local_samples + args.batch_size - 1) // args.batch_size

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"model.freqs_cos", "model.freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), args.batch_size, start_step
            )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            Logger(
                f"DPO Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
            )
            train_epoch(epoch, loader, steps_per_epoch, start_step, wandb)
        else:
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            train_epoch(epoch, loader, steps_per_epoch, 0, wandb)


if __name__ == "__main__":
    main()
