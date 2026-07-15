from __future__ import annotations

import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from .a1_5_data import ANSWER_LABELS, REGISTER_NAMES, load_a15_records
from .model import load_qwen35_text_only


FINAL_RE = re.compile(
    r"FINAL(?:\s+ANSWER)?\s*:\s*(?:X\s*=\s*)?([A-J])\b",
    re.IGNORECASE,
)
PROSE_FINAL_RE = re.compile(
    r"(?:final\s+answer|answer|symbol(?:\s+in[^\n.!?]{0,96})?)\s*(?:is|=|:)\s*\**([A-J])\b",
    re.IGNORECASE,
)
FINAL_STATE_MARKER_RE = re.compile(
    r"(?:final\s+state|final\s+trace|end\s+state)\b",
    re.IGNORECASE,
)
FINAL_WORD_RE = re.compile(r"\bfinal\b", re.IGNORECASE)
STATE_RE = re.compile(
    r"Step\s*(\d+)\s*:\s*amber\s*=\s*([A-J])\s*,\s*cobalt\s*=\s*([A-J])\s*,\s*jade\s*=\s*([A-J])",
    re.IGNORECASE,
)
STATE_FALLBACK_RE = re.compile(
    r"State\s*:\s*amber\s*=\s*([A-J])\s*,\s*cobalt\s*=\s*([A-J])\s*,\s*jade\s*=\s*([A-J])",
    re.IGNORECASE,
)


def parse_a15_text(completion: str) -> tuple[str | None, dict[int, list[str]]]:
    final_match = list(FINAL_RE.finditer(completion))
    prose_match = list(PROSE_FINAL_RE.finditer(completion))
    candidates = final_match + prose_match
    answer = candidates[-1].group(1).upper() if candidates else None
    states = {
        int(match.group(1)): [match.group(2).upper(), match.group(3).upper(), match.group(4).upper()]
        for match in STATE_RE.finditer(completion)
    }
    fallback_states = [
        [match.group(1).upper(), match.group(2).upper(), match.group(3).upper()]
        for match in STATE_FALLBACK_RE.finditer(completion)
    ]
    for index, state in enumerate(fallback_states, start=1):
        states.setdefault(index, state)
    return answer, states


def _prompt(record: dict[str, Any], demonstrations: Sequence[dict[str, Any]]) -> str:
    instruction = (
        "Track the three registers exactly. First write one Start line, then one line for every operation in order "
        "using `Step n: amber=X, cobalt=Y, jade=Z.`, and finish with `FINAL: X`. "
        "Do not skip or reorder steps.\n\n"
    )
    demo_text = ""
    for demo in demonstrations:
        demo_text += "EXAMPLE QUESTION:\n" + demo["question"] + "\nEXAMPLE TRACE:\n"
        demo_text += "Start: amber={amber}, cobalt={cobalt}, jade={jade}.\n".format(**demo["start_state"])
        for index, state in enumerate(demo["state_trajectory"], start=1):
            demo_text += (
                f"Step {index}: amber={state['amber']}, cobalt={state['cobalt']}, jade={state['jade']}.\n"
            )
        demo_text += f"FINAL: {demo['answer']}\n\n"
    return instruction + demo_text + "QUESTION:\n" + record["question"] + "\nANSWER:\n"


class A15AnswerCriteria(StoppingCriteria):
    def __init__(self, tokenizer: Any, prompt_length: int, required_steps: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.required_steps = required_steps

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> torch.BoolTensor:
        return torch.tensor(
            [self._is_terminal(self.tokenizer.decode(row[self.prompt_length :], skip_special_tokens=True)) for row in input_ids],
            dtype=torch.bool,
            device=input_ids.device,
        )

    def _is_terminal(self, text: str) -> bool:
        if (
            FINAL_RE.search(text)
            or PROSE_FINAL_RE.search(text)
            or FINAL_STATE_MARKER_RE.search(text)
            or FINAL_WORD_RE.search(text)
        ):
            return True
        if self.required_steps is None:
            return False
        explicit_steps = {int(match.group(1)) for match in STATE_RE.finditer(text)}
        fallback_steps = len(STATE_FALLBACK_RE.findall(text))
        return len(explicit_steps) >= self.required_steps or fallback_steps >= self.required_steps


def _generate_no_cap(
    model: Any,
    tokenizer: Any,
    encoded: dict[str, torch.Tensor],
    *,
    required_steps: int | None = None,
    # Keep transport chunks small enough that the explicit wall-time safety
    # check is observable even when one generate call is slow. This is not a
    # total output cap: the loop continues until a semantic stop or context
    # boundary.
    chunk_tokens: int = 16,
    max_wall_seconds: float | None = None,
) -> tuple[torch.Tensor, str]:
    prompt_length = int(encoded["input_ids"].shape[1])
    sequence = encoded["input_ids"]
    attention = encoded["attention_mask"]
    context_limit = int(tokenizer.model_max_length)
    started = time.perf_counter()
    termination_reason = "physical_context_boundary"
    while sequence.shape[1] < context_limit:
        if max_wall_seconds is not None and time.perf_counter() - started >= max_wall_seconds:
            termination_reason = "safety_wall_timeout"
            break
        chunk = min(chunk_tokens, context_limit - int(sequence.shape[1]))
        generated = model.generate(
            input_ids=sequence,
            attention_mask=attention,
            max_new_tokens=chunk,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList(
                [A15AnswerCriteria(tokenizer, prompt_length, required_steps=required_steps)]
            ),
        )
        new_tokens = generated[:, sequence.shape[1] :]
        sequence = generated
        if new_tokens.numel() == 0:
            termination_reason = "eos_or_empty"
            break
        completion = tokenizer.decode(sequence[0, prompt_length:], skip_special_tokens=True)
        if max_wall_seconds is not None and time.perf_counter() - started >= max_wall_seconds:
            termination_reason = "safety_wall_timeout"
            break
        if (
            FINAL_RE.search(completion)
            or PROSE_FINAL_RE.search(completion)
            or FINAL_STATE_MARKER_RE.search(completion)
            or FINAL_WORD_RE.search(completion)
            or (
                required_steps is not None
                and (
                    len({int(match.group(1)) for match in STATE_RE.finditer(completion)}) >= required_steps
                    or len(STATE_FALLBACK_RE.findall(completion)) >= required_steps
                )
            )
        ):
            termination_reason = "semantic_terminal"
            break
        if tokenizer.eos_token_id is not None and bool((new_tokens == tokenizer.eos_token_id).any()):
            termination_reason = "eos"
            break
        attention = torch.ones_like(sequence, dtype=attention.dtype)
    return sequence, termination_reason


def run_a15_text_baseline(
    *,
    data_dir: Path,
    output_path: Path,
    split: str,
    max_examples: int,
    shots: int,
    model_id: str = "Qwen/Qwen3.5-2B",
    revision: str = "15852e8c16360a2fea060d615a32b45270f8a8fc",
    device: str = "cuda",
    seed: int = 20260713,
    max_input_tokens: int = 512,
    max_wall_seconds: float | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = load_qwen35_text_only(model_id, revision, dtype=dtype, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    records = load_a15_records(data_dir, split)[:max_examples]
    demonstrations = load_a15_records(data_dir, "train")[:shots]
    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    input_tokens = 0
    generated_tokens = 0
    for record in records:
        prompt = _prompt(record, demonstrations)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=False).to(device)
        input_length = int(encoded["input_ids"].shape[1])
        if input_length > max_input_tokens:
            raise ValueError(f"input length {input_length} exceeds max_input_tokens={max_input_tokens}; refusing truncation")
        started = time.perf_counter()
        with torch.inference_mode():
            generated, termination_reason = _generate_no_cap(
                model,
                tokenizer,
                encoded,
                required_steps=len(record["operations"]),
                max_wall_seconds=max_wall_seconds,
            )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        latency = time.perf_counter() - started
        completion_ids = generated[0, input_length:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        prediction, parsed_states = parse_a15_text(completion)
        derived_from_trace = False
        if prediction is None:
            terminal_state = parsed_states.get(len(record["operations"]))
            if terminal_state is not None:
                prediction = terminal_state[REGISTER_NAMES.index(record["query_register"])]
                derived_from_trace = True
        expected_states = {
            index: [state[name] for name in REGISTER_NAMES]
            for index, state in enumerate(record["state_trajectory"], start=1)
        }
        state_exact = bool(expected_states) and all(parsed_states.get(index) == state for index, state in expected_states.items())
        predictions.append(
            {
                "example_id": record["example_id"],
                "target": record["answer"],
                "prediction": prediction,
                "correct": prediction == record["answer"],
                "parsed": prediction is not None,
                "final_derived_from_trace": derived_from_trace,
                "termination_reason": termination_reason,
                "state_full_exact": state_exact,
                "input_tokens": input_length,
                "generated_reasoning_tokens": int(completion_ids.numel()),
                "latency_seconds": latency,
                "completion": completion,
            }
        )
        input_tokens += input_length
        generated_tokens += int(completion_ids.numel())
        latencies.append(latency)
        if device.startswith("cuda"):
            # Generation uses a variable-size KV cache.  Release completed
            # example tensors so the no-artificial-cap probe does not turn
            # allocator caching into a false OOM signal.
            del generated, encoded, completion_ids
            torch.cuda.empty_cache()
        if (len(predictions) % 8 == 0) or (len(predictions) == len(records)):
            print(
                f"[text-baseline] processed {len(predictions)}/{len(records)}",
                file=sys.stderr,
                flush=True,
            )
            # Keep an auditable partial record if a long no-cap probe is
            # externally interrupted or the host runs out of memory. The
            # final JSON is still written only after all requested examples.
            partial_path = output_path.with_suffix(output_path.suffix + ".partial.json")
            partial_path.write_text(
                __import__("json").dumps(
                    {
                        "schema_version": "yggdrasil.v2-a1.5.text-baseline.partial.v1",
                        "data": {"split": split, "examples_requested": len(records), "examples_completed": len(predictions), "shots": shots},
                        "generation": {"artificial_output_token_cap": None, "transport_chunk_tokens": 16, "max_wall_seconds_per_example": max_wall_seconds},
                        "predictions": predictions,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "yggdrasil.v2-a1.5.text-baseline.v1",
        "evidence_level": "probe" if max_examples > 32 else "smoke",
        "model": {"model_id": model_id, "revision": revision, "dtype": str(next(model.parameters()).dtype)},
        "data": {"schema": "yggdrasil.v2-a1.5.symbolic-state-machine.v1", "split": split, "examples": len(records), "shots": shots},
        "generation": {
            "do_sample": False,
            "artificial_output_token_cap": None,
            "termination": "semantic FINAL, complete state trace, EOS, or physical context boundary",
            "transport_chunk_tokens": 16,
            "max_input_tokens_guard": max_input_tokens,
            "max_wall_seconds_per_example": max_wall_seconds,
        },
        "quality": {
            "final_answer_accuracy": sum(item["correct"] for item in predictions) / max(1, len(predictions)),
            "parse_rate": sum(item["parsed"] for item in predictions) / max(1, len(predictions)),
            "state_full_exact": sum(item["state_full_exact"] for item in predictions) / max(1, len(predictions)),
        },
        "cost": {
            "input_tokens": input_tokens,
            "generated_reasoning_tokens": generated_tokens,
            "latency_seconds_total": sum(latencies),
            "latency_seconds_median": statistics.median(latencies) if latencies else 0.0,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0,
            "flops": "not measured; no quality-cost claim allowed",
        },
        "predictions": predictions,
    }
    output_path.write_text(__import__("json").dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
