import copy
import os

import torch
from torch import nn


def safe_torch_load(path, map_location="cpu", weights_only=True):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=None, dropout=0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear only supports nn.Linear base layers")
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out

    def merged_linear(self):
        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
        )
        merged.weight.data.copy_(self.base_layer.weight.data)
        if self.base_layer.bias is not None:
            merged.bias.data.copy_(self.base_layer.bias.data)
        delta = (self.lora_B.weight.data @ self.lora_A.weight.data) * self.scaling
        merged.weight.data.add_(delta.to(merged.weight.dtype))
        return merged


def _unwrap_model(model):
    raw_model = model
    while True:
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
            continue
        if hasattr(raw_model, "module"):
            raw_model = raw_model.module
            continue
        break
    return raw_model


def _normalize_targets(target_modules):
    if target_modules is None:
        return ("q_proj", "k_proj", "v_proj", "o_proj")
    if isinstance(target_modules, str):
        return tuple(
            target.strip() for target in target_modules.split(",") if target.strip()
        )
    return tuple(target_modules)


def apply_lora(model, rank=8, alpha=None, dropout=0.0, target_modules=None):
    raw_model = _unwrap_model(model)
    targets = _normalize_targets(target_modules)
    replaced = []

    def _replace(module, prefix=""):
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(
                    module,
                    child_name,
                    LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
                )
                replaced.append(full_name)
                continue
            _replace(child, full_name)

    _replace(raw_model)
    return replaced


def mark_only_lora_trainable(model):
    raw_model = _unwrap_model(model)
    lora_params = []
    for name, param in raw_model.named_parameters():
        is_lora_param = "lora_A" in name or "lora_B" in name
        param.requires_grad = is_lora_param
        if is_lora_param:
            lora_params.append(param)
    return lora_params


def lora_state_dict(model):
    raw_model = _unwrap_model(model)
    state_dict = raw_model.state_dict()
    return {
        key: value.detach().cpu()
        for key, value in state_dict.items()
        if "lora_A" in key or "lora_B" in key
    }


def save_lora(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(lora_state_dict(model), path)


def load_lora(model, path, map_location="cpu"):
    raw_model = _unwrap_model(model)
    state_dict = safe_torch_load(path, map_location=map_location, weights_only=True)
    cleaned_state = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("_orig_mod."):
            key = key[10:]
        cleaned_state[key] = value
    return raw_model.load_state_dict(cleaned_state, strict=False)


def merge_lora(model, lora_path=None, save_path=None):
    raw_model = _unwrap_model(model)
    if lora_path is not None:
        load_lora(raw_model, lora_path)

    merged_model = copy.deepcopy(raw_model)

    def _merge(module):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, child_name, child.merged_linear())
                continue
            _merge(child)

    _merge(merged_model)

    if save_path is not None:
        torch.save(
            {key: value.cpu().half() for key, value in merged_model.state_dict().items()},
            save_path,
        )
    return merged_model
