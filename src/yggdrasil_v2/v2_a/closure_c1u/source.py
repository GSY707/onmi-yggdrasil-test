from __future__ import annotations

"""Pinned offline Qwen source audit and literal one-card encoder."""

import gc
import platform
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping

import torch

from . import contract
from .artifacts import sha256_file
from yggdrasil_v2.v2_a.closure_c1.cache import (
    load_qwen35_text_only,
    load_qwen35_tokenizer,
)


def audit_source_assets(
    *, resolver: Callable[..., str] | None = None
) -> dict[str, Any]:
    """Hash every frozen local source asset; network fallback is forbidden."""

    if resolver is None:
        from huggingface_hub import hf_hub_download

        resolver = hf_hub_download
    rows: dict[str, Any] = {}
    for filename, expected in contract.SOURCE_ASSETS.items():
        try:
            path = Path(
                resolver(
                    repo_id=contract.SOURCE_MODEL_ID,
                    filename=filename,
                    revision=contract.SOURCE_MODEL_REVISION,
                    local_files_only=True,
                )
            )
            observed_bytes = path.stat().st_size
            observed_sha = sha256_file(path)
            checks = {
                "exists": path.is_file(),
                "bytes": observed_bytes == int(expected["bytes"]),
                "sha256": observed_sha == str(expected["sha256"]),
            }
            rows[filename] = {
                "passed": all(checks.values()),
                "checks": checks,
                "bytes": observed_bytes,
                "sha256": observed_sha,
                "path_name": path.name,
            }
        except Exception as exc:
            rows[filename] = {
                "passed": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.source-assets-audit.v1",
        "model_id": contract.SOURCE_MODEL_ID,
        "revision": contract.SOURCE_MODEL_REVISION,
        "local_files_only": True,
        "rows": rows,
        "passed": len(rows) == len(contract.SOURCE_ASSETS)
        and all(row.get("passed") is True for row in rows.values()),
    }


def audit_device(device: str | torch.device = "cuda") -> dict[str, Any]:
    """Fail-closed runtime/device audit for the registered 4070 BF16 path."""

    import transformers

    selected = torch.device(device)
    cuda_available = torch.cuda.is_available()
    count = torch.cuda.device_count() if cuda_available else 0
    device_index = 0 if selected.index is None else int(selected.index)
    name = torch.cuda.get_device_name(device_index) if cuda_available and count > device_index else ""
    capability = (
        torch.cuda.get_device_capability(device_index)
        if cuda_available and count > device_index
        else (0, 0)
    )
    bf16 = bool(torch.cuda.is_bf16_supported()) if cuda_available else False
    free_bytes, total_bytes = (
        torch.cuda.mem_get_info(device_index)
        if cuda_available and count > device_index
        else (0, 0)
    )
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks = {
        "selected_cuda": selected.type == "cuda",
        "cuda_available": cuda_available,
        "device_count": count == contract.DEVICE_COUNT,
        "device_index": device_index == 0,
        "device_name": contract.DEVICE_NAME_FRAGMENT.casefold() in name.casefold(),
        "compute_capability": tuple(capability) >= tuple(contract.MINIMUM_COMPUTE_CAPABILITY),
        "bf16_supported": bf16,
        "free_cuda_bytes": int(free_bytes) >= contract.MINIMUM_FREE_CUDA_BYTES,
        "python_version": python_version == contract.PYTHON_VERSION,
        "torch_version": str(torch.__version__) == contract.TORCH_VERSION,
        "cuda_version": str(torch.version.cuda) == contract.CUDA_VERSION,
        "transformers_version": str(transformers.__version__) == contract.TRANSFORMERS_VERSION,
    }
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.device-audit.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "selected_device": str(selected),
        "device_count_observed": count,
        "device_name": name,
        "compute_capability": list(capability),
        "bf16_supported": bf16,
        "free_cuda_bytes": int(free_bytes),
        "total_cuda_bytes": int(total_bytes),
        "python": python_version,
        "python_implementation": platform.python_implementation(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "transformers": str(transformers.__version__),
    }


class QwenCardEncoder:
    """A callable whose every invocation performs one source-model forward."""

    def __init__(self, *, device: str | torch.device = "cuda") -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("registered C1U source encoder requires CUDA")
        self.tokenizer = load_qwen35_tokenizer(
            contract.SOURCE_MODEL_ID,
            contract.SOURCE_MODEL_REVISION,
            local_files_only=True,
        )
        self.model = load_qwen35_text_only(
            contract.SOURCE_MODEL_ID,
            contract.SOURCE_MODEL_REVISION,
            dtype=torch.float16,
            device=self.device,
            local_files_only=True,
        )
        if type(self.model).__name__ != contract.SOURCE_MODEL_CLASS:
            raise TypeError(f"unexpected source model class: {type(self.model).__name__}")
        if type(self.tokenizer).__name__ != contract.SOURCE_TOKENIZER_CLASS:
            raise TypeError(f"unexpected tokenizer class: {type(self.tokenizer).__name__}")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("source model parameters must remain frozen")
        self.calls = 0
        self.token_counts: list[int] = []
        self.started_at = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(self.device)

    def __call__(self, text: str) -> Mapping[str, torch.Tensor]:
        if type(text) is not str or not text:
            raise ValueError("card encoder requires one non-empty string")
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise RuntimeError("one-card encoder received a non-singleton token batch")
        tokens = int(input_ids.shape[1])
        if tokens < 1 or tokens > contract.SOURCE_MAX_CARD_TOKENS:
            raise ValueError(
                f"card token count {tokens} outside [1,{contract.SOURCE_MAX_CARD_TOKENS}]"
            )
        mask = encoded.get("attention_mask", torch.ones_like(input_ids)).bool()
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=mask.to(self.device),
                use_cache=False,
            )
        hidden = output.last_hidden_state
        if hidden.shape != (1, tokens, contract.SOURCE_HIDDEN_WIDTH):
            raise RuntimeError(f"unexpected source hidden shape: {tuple(hidden.shape)}")
        if hidden.dtype != torch.float16 or not bool(torch.isfinite(hidden).all()):
            raise RuntimeError("source hidden must be finite FP16")
        self.calls += 1
        self.token_counts.append(tokens)
        return {
            "hidden": hidden[0].detach().cpu().contiguous(),
            "mask": mask[0].detach().cpu().contiguous(),
            "token_ids": input_ids[0].detach().cpu().to(torch.int64).contiguous(),
        }

    def report(self) -> dict[str, Any]:
        counts = list(self.token_counts)
        parameters = sum(parameter.numel() for parameter in self.model.parameters())
        return {
            "schema_version": f"{contract.SCHEMA_PREFIX}.source-encoder-report.v1",
            "model_id": contract.SOURCE_MODEL_ID,
            "revision": contract.SOURCE_MODEL_REVISION,
            "model_class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "calls": self.calls,
            "one_card_per_forward_call": True,
            "source_parameters": int(parameters),
            "trainable_source_parameters": sum(
                parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
            ),
            "hidden_width": contract.SOURCE_HIDDEN_WIDTH,
            "hidden_dtype": contract.SOURCE_DTYPE,
            "minimum_tokens": min(counts) if counts else 0,
            "maximum_tokens": max(counts) if counts else 0,
            "total_tokens": sum(counts),
            "elapsed_seconds": time.perf_counter() - self.started_at,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(self.device)),
        }

    def release(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


__all__ = ["QwenCardEncoder", "audit_device", "audit_source_assets"]
