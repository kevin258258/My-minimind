import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def safe_torch_load(path, map_location="cpu", weights_only=True):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        # 兼容较老版本PyTorch（不支持weights_only参数）
        return torch.load(path, map_location=map_location)


# 检查是否是主进程
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# 日志
def Logger(content):
    if is_main_process():
        print(content)


# 动态学习率计算
def get_lr(current_step, total_steps, lr):
    return (
        lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
    )  # ！修正：原公式 step=0 时 lr=1.1*lr 超出设定值，现修正为 step=0→lr, step=end→0.1*lr


# 初始化分布式
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


# 设置种子
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 设置检查点
def lm_checkpoint(
    lm_config,
    weight="full_sft",
    model=None,
    optimizer=None,
    epoch=0,
    step=0,
    wandb=None,
    save_dir="checkpoints",
    save_full_model=True,
    **kwargs,
):
    os.makedirs(save_dir, exist_ok=True)

    moe_path = "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth"
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth"

    if model is not None:
        from torch.nn.parallel import DistributedDataParallel

        state_dict = kwargs.pop("model_state_dict", None)
        if state_dict is None:
            if isinstance(model, DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

        if save_full_model:
            ckp_tmp = ckp_path + ".tmp"
            torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
            os.replace(ckp_tmp, ckp_path)

        wandb_id = None
        if wandb:
            run = getattr(wandb, "run", None)
            wandb_id = getattr(run, "id", None) if run is not None else None

        resume_data = {
            "model": state_dict if save_full_model else None,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "wandb_id": wandb_id,
        }

        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "state_dict"):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + ".tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = safe_torch_load(
                resume_path, map_location="cpu", weights_only=False
            )
            saved_ws = ckp_data.get("world_size", 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1

            if saved_ws != current_ws:
                ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
                Logger(
                    f"GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data['step']}"
                )

            return ckp_data
        return None


# 初始化模型
def init_model(
    lm_config,
    from_weight="pretrain",
    tokenizer_path=None,
    save_dir="../out",
    device="cuda",
):
    from transformers import AutoTokenizer
    from model.model import HollowStoneMindForCausalLM

    if tokenizer_path is None:
        raise ValueError(
            "tokenizer_path is required. Pass a directory that contains a Hugging Face tokenizer."
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model = HollowStoneMindForCausalLM(lm_config)

    if from_weight != "none":
        moe_suffix = (
            "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
        )
        weight_path = (
            f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
        )

        weights = safe_torch_load(weight_path, map_location=device, weights_only=True)

        model.load_state_dict(weights, strict=False)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f"所加载Model可训练参数：{total_params / 1e6:.3f} 百万")

    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler  #
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []  # 当前批次
        skipped = 0  # 已跳过的批次数

        for idx in self.sampler:
            batch.append(idx)  # 添加样本到当前批次

            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1  # 增加跳过计数
                    batch = []  # 清空批次，不返回
                    continue  # 跳过这个批次

                yield batch
                batch = []  # 重置批次

        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size

        return max(0, total_batches - self.skip_batches)
