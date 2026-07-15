from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from .artifacts import environment_record, write_json
from .data import DATA_SCHEMA_VERSION, load_jsonl
from .model import load_qwen35_text_only, text_decoder
from .prompts import build_messages, has_complete_answer, parse_answer


class AnswerCompleteCriteria(StoppingCriteria):
    """Stop on a complete task answer, never on an arbitrary token budget."""

    def __init__(self, tokenizer: Any, prompt_length: int, mode: str) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.mode = mode

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> torch.BoolTensor:
        completed: list[bool] = []
        for row in input_ids:
            text = self.tokenizer.decode(row[self.prompt_length :], skip_special_tokens=True)
            completed.append(has_complete_answer(text, self.mode))
        return torch.tensor(completed, dtype=torch.bool, device=input_ids.device)


def _load_backbone(model_id: str, revision: str, device: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    if config.model_type == "qwen3_5":
        model = load_qwen35_text_only(
            model_id,
            revision,
            dtype=dtype,
            device=device,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
    if config.model_type != "qwen3_5":
        model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]


def _generate_until_complete(
    *,
    model: Any,
    tokenizer: Any,
    encoded: dict[str, torch.Tensor],
    mode: str,
    transport_chunk_tokens: int = 128,
) -> torch.LongTensor:
    """Generate without a total output cap, using bounded transport chunks.

    A context-sized max_length makes Qwen3.5 reserve pathological cache/mask
    state on an 8 GB GPU. Chunking is an execution detail, not an output cap:
    generation continues until a semantic answer, EOS, or context exhaustion.
    """

    input_length = int(encoded["input_ids"].shape[1])
    sequence = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    context_limit = int(tokenizer.model_max_length)
    while sequence.shape[1] < context_limit:
        chunk_size = min(transport_chunk_tokens, context_limit - int(sequence.shape[1]))
        generated = model.generate(
            input_ids=sequence,
            attention_mask=attention_mask,
            max_new_tokens=chunk_size,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList(
                [AnswerCompleteCriteria(tokenizer, input_length, mode)]
            ),
        )
        new_tokens = generated[:, sequence.shape[1] :]
        sequence = generated
        if new_tokens.numel() == 0:
            break
        completion = tokenizer.decode(sequence[0, input_length:], skip_special_tokens=True)
        if has_complete_answer(completion, mode):
            break
        if tokenizer.eos_token_id is not None and bool((new_tokens == tokenizer.eos_token_id).any()):
            break
        attention_mask = torch.ones_like(sequence, dtype=encoded["attention_mask"].dtype)
    return sequence


def run_text_baselines(
    *,
    data_dir: Path,
    output_path: Path,
    model_id: str,
    revision: str,
    split: str,
    max_examples: int,
    seed: int,
    device: str,
    modes: tuple[str, ...] = ("direct", "answer_only", "text_cot"),
    enable_thinking: bool = False,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    tokenizer, model = _load_backbone(model_id, revision, device)
    records = load_jsonl(data_dir / f"{split}.jsonl")[:max_examples]
    demonstrations = load_jsonl(data_dir / "train.jsonl")[:2]
    mode_results: dict[str, Any] = {}

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    for mode in modes:
        predictions: list[dict[str, Any]] = []
        latencies: list[float] = []
        generated_tokens = 0
        input_tokens = 0
        for record in records:
            messages = build_messages(record, mode, demonstrations if mode != "direct" else [])
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            encoded = tokenizer(prompt, return_tensors="pt").to(device)
            input_length = int(encoded["input_ids"].shape[1])
            started = time.perf_counter()
            with torch.inference_mode():
                generated = _generate_until_complete(
                    model=model,
                    tokenizer=tokenizer,
                    encoded=encoded,
                    mode=mode,
                )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            latency = time.perf_counter() - started
            completion_ids = generated[0, input_length:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            prediction = parse_answer(completion)
            token_count = int(completion_ids.numel())
            input_tokens += input_length
            generated_tokens += token_count
            latencies.append(latency)
            predictions.append(
                {
                    "example_id": record["example_id"],
                    "target": record["answer"],
                    "prediction": prediction,
                    "correct": prediction == record["answer"],
                    "parsed": prediction is not None,
                    "input_tokens": input_length,
                    "generated_tokens": token_count,
                    "latency_seconds": latency,
                    "completion": completion,
                }
            )

        exact = sum(item["correct"] for item in predictions) / max(1, len(predictions))
        parse_rate = sum(item["parsed"] for item in predictions) / max(1, len(predictions))
        mode_results[mode] = {
            "exact_match": exact,
            "parse_rate": parse_rate,
            "examples": len(predictions),
            "input_tokens": input_tokens,
            "generated_reasoning_tokens": generated_tokens,
            "autogressive_sampling_steps": generated_tokens,
            "latency_seconds_total": sum(latencies),
            "latency_seconds_median": statistics.median(latencies) if latencies else 0.0,
            "latency_seconds_p95": _percentile(latencies, 0.95),
            "predictions": predictions,
        }

    result = {
        "schema_version": "yggdrasil.v2-a.results.v1",
        "evidence_level": "smoke" if max_examples <= 32 else "probe",
        "experiment": "V2-A0 text baselines",
        "data_schema_version": DATA_SCHEMA_VERSION,
        "model": {
            "model_id": model_id,
            "revision": revision,
            "license": "apache-2.0",
            "dtype": str(next(model.parameters()).dtype),
        },
        "split": split,
        "seed": seed,
        "generation": {
            "do_sample": False,
            "temperature": None,
            "termination": "semantic final answer, EOS, or physical context boundary",
            "artificial_output_token_cap": None,
            "transport_chunk_tokens": 128,
            "enable_thinking": enable_thinking,
        },
        "quality": mode_results,
        "parameters": {
            "frozen_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "active_text_parameters": sum(parameter.numel() for parameter in text_decoder(model).parameters()),
            "trainable_parameters": 0,
            "active_parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "cost": {
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0,
            "kv_cache_estimate": "reported after A0 prompt-length calibration; not inferred as measured",
            "flops": "not yet measured; no quality-cost claim allowed",
            "output_limit": "semantic final answer, EOS, or model context only; no max_new_tokens cap",
        },
        "environment": environment_record(torch),
        "boundaries": [
            "Deterministic local baseline only; not a multi-seed formal result.",
            "Text CoT correctness is scored by final answer, not by trace faithfulness.",
            "No latent architecture claim follows from this artifact.",
        ],
    }
    write_json(output_path, result)
    return result
