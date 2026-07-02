from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import time
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


Mode = Literal["direct", "visible", "latent"]


GRID_SIZE = 8
MOVE_COUNT = 6
MOVES = ("U", "D", "L", "R")
SPECIAL = ("<pad>", "<bos>", "<eos>", "<latent>", "Q", "M", "A", "T")
TOKENS = (
    *SPECIAL,
    *(f"X{i}" for i in range(GRID_SIZE)),
    *(f"Y{i}" for i in range(GRID_SIZE)),
    *MOVES,
)
TOKEN_TO_ID = {token: index for index, token in enumerate(TOKENS)}
ID_TO_TOKEN = {index: token for token, index in TOKEN_TO_ID.items()}
LATENT_ID = TOKEN_TO_ID["<latent>"]
EOS_ID = TOKEN_TO_ID["<eos>"]


@dataclass(frozen=True)
class Example:
    x: int
    y: int
    moves: tuple[str, ...]


@dataclass(frozen=True)
class TrainConfig:
    move_count: int = 6
    train_size: int = 4096
    val_size: int = 1024
    test_size: int = 2048
    batch_size: int = 128
    text_steps: int = 1000
    direct_steps: int = 1000
    latent_steps: int = 700
    latent_scratch_steps: int = 700
    eval_every: int = 250
    seed: int = 20260701
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.0
    lr: float = 7e-4


class CausalBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        attn_input = self.ln_1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, attn_mask=causal_mask, need_weights=False)
        x = x + attn_output
        return x + self.ff(self.ln_2(x))


class TinyCausalLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(CausalBlock(d_model, n_heads, dropout) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        for block in self.blocks:
            x = block(x, causal_mask)
        return self.head(self.ln_f(x))


def apply_move(x: int, y: int, move: str) -> tuple[int, int]:
    if move == "U":
        return x, (y - 1) % GRID_SIZE
    if move == "D":
        return x, (y + 1) % GRID_SIZE
    if move == "L":
        return (x - 1) % GRID_SIZE, y
    if move == "R":
        return (x + 1) % GRID_SIZE, y
    raise ValueError(f"unknown move: {move}")


def trace(example: Example) -> list[tuple[int, int]]:
    x, y = example.x, example.y
    states: list[tuple[int, int]] = []
    for move in example.moves:
        x, y = apply_move(x, y, move)
        states.append((x, y))
    return states


def final_state(example: Example) -> tuple[int, int]:
    return trace(example)[-1]


def generate_examples(count: int, *, seed: int, exclude: set[tuple[int, int, tuple[str, ...]]] | None = None) -> list[Example]:
    rng = random.Random(seed)
    seen = set(exclude or set())
    examples: list[Example] = []
    while len(examples) < count:
        item = (
            rng.randrange(GRID_SIZE),
            rng.randrange(GRID_SIZE),
            tuple(rng.choice(MOVES) for _ in range(MOVE_COUNT)),
        )
        if item in seen:
            continue
        seen.add(item)
        examples.append(Example(x=item[0], y=item[1], moves=item[2]))
    return examples


def example_key(example: Example) -> tuple[int, int, tuple[str, ...]]:
    return example.x, example.y, example.moves


def prompt_tokens(example: Example) -> list[str]:
    return ["<bos>", "Q", f"X{example.x}", f"Y{example.y}", "M", *example.moves]


def target_tokens(example: Example, mode: Mode) -> tuple[list[str], int]:
    final_x, final_y = final_state(example)
    if mode == "direct":
        target = ["A", f"X{final_x}", f"Y{final_y}", "<eos>"]
        return target, len(prompt_tokens(example))
    if mode == "visible":
        target: list[str] = []
        for x, y in trace(example):
            target.extend(["T", f"X{x}", f"Y{y}"])
        target.extend(["A", f"X{final_x}", f"Y{final_y}", "<eos>"])
        return target, len(prompt_tokens(example))
    if mode == "latent":
        target = ["<latent>"] * MOVE_COUNT + ["A", f"X{final_x}", f"Y{final_y}", "<eos>"]
        return target, len(prompt_tokens(example)) + MOVE_COUNT
    raise ValueError(f"unknown mode: {mode}")


def sequence_and_label(example: Example, mode: Mode) -> tuple[list[int], list[int]]:
    prompt = prompt_tokens(example)
    target, first_supervised_index = target_tokens(example, mode)
    sequence = [TOKEN_TO_ID[token] for token in [*prompt, *target]]
    labels = sequence[1:]
    masked_labels = [
        label if index + 1 >= first_supervised_index else -100
        for index, label in enumerate(labels)
    ]
    return sequence[:-1], masked_labels


def max_sequence_length() -> int:
    probe = Example(0, 0, ("U",) * MOVE_COUNT)
    return max(len(sequence_and_label(probe, mode)[0]) for mode in ("direct", "visible", "latent"))


def make_batch(
    examples: list[Example],
    mode: Mode,
    *,
    rng: random.Random,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = [examples[rng.randrange(len(examples))] for _ in range(batch_size)]
    xs, ys = zip(*(sequence_and_label(example, mode) for example in batch))
    return (
        torch.tensor(xs, dtype=torch.long, device=device),
        torch.tensor(ys, dtype=torch.long, device=device),
    )


def clone_model(model: TinyCausalLM) -> TinyCausalLM:
    copied = TinyCausalLM(
        vocab_size=len(TOKENS),
        max_seq_len=model.max_seq_len,
        d_model=model.token_embedding.embedding_dim,
        n_layers=len(model.blocks),
        n_heads=model.blocks[0].attn.num_heads,
        dropout=0.0,
    )
    copied.load_state_dict({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
    return copied


def new_model(config: TrainConfig, device: torch.device) -> TinyCausalLM:
    model = TinyCausalLM(
        vocab_size=len(TOKENS),
        max_seq_len=max_sequence_length() + 8,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        dropout=config.dropout,
    )
    return model.to(device)


def train_model(
    model: TinyCausalLM,
    train_examples: list[Example],
    val_examples: list[Example],
    *,
    mode: Mode,
    config: TrainConfig,
    steps: int,
    seed: int,
    device: torch.device,
    label: str,
) -> dict[str, object]:
    model.train()
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        x, y = make_batch(train_examples, mode, rng=rng, batch_size=config.batch_size, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % config.eval_every == 0 or step == steps:
            val_metrics = evaluate(model, val_examples[: min(len(val_examples), 512)], mode=mode, device=device)
            history.append(
                {
                    "step": step,
                    "train_loss": round(float(loss.detach().cpu()), 4),
                    "val_exact": round(val_metrics["exact_accuracy"], 4),
                }
            )
            print(
                f"{label:18s} step={step:4d} loss={float(loss.detach().cpu()):.4f} "
                f"val_exact={val_metrics['exact_accuracy']:.3f}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "label": label,
        "mode": mode,
        "steps": steps,
        "seconds": round(time.perf_counter() - started, 2),
        "history": history,
    }


@torch.no_grad()
def generate_answer_tokens(model: TinyCausalLM, example: Example, *, mode: Mode, device: torch.device) -> list[str]:
    model.eval()
    if mode == "latent":
        tokens = prompt_tokens(example) + ["<latent>"] * MOVE_COUNT
        max_new_tokens = 4
    elif mode == "visible":
        tokens = prompt_tokens(example)
        max_new_tokens = MOVE_COUNT * 3 + 4
    else:
        tokens = prompt_tokens(example)
        max_new_tokens = 4

    input_ids = torch.tensor([[TOKEN_TO_ID[token] for token in tokens]], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_id = int(torch.argmax(logits[0, -1]).detach().cpu())
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )
        if next_id == EOS_ID:
            break
    return [ID_TO_TOKEN[int(item)] for item in input_ids[0].detach().cpu().tolist()]


def parse_answer(tokens: list[str]) -> tuple[int, int] | None:
    for index, token in enumerate(tokens):
        if token != "A" or index + 2 >= len(tokens):
            continue
        x_token = tokens[index + 1]
        y_token = tokens[index + 2]
        if x_token.startswith("X") and y_token.startswith("Y"):
            try:
                return int(x_token[1:]), int(y_token[1:])
            except ValueError:
                return None
    return None


@torch.no_grad()
def evaluate(
    model: TinyCausalLM,
    examples: list[Example],
    *,
    mode: Mode,
    device: torch.device,
) -> dict[str, float | int]:
    correct = 0
    parseable = 0
    generated_lengths: list[int] = []
    prompt_len = len(prompt_tokens(examples[0])) if examples else 0
    for example in examples:
        tokens = generate_answer_tokens(model, example, mode=mode, device=device)
        answer = parse_answer(tokens)
        if answer is not None:
            parseable += 1
        if answer == final_state(example):
            correct += 1
        generated_lengths.append(max(0, len(tokens) - prompt_len))
    return {
        "count": len(examples),
        "exact_accuracy": correct / len(examples) if examples else 0.0,
        "parseable_rate": parseable / len(examples) if examples else 0.0,
        "avg_generated_or_inserted_tokens": sum(generated_lengths) / len(generated_lengths)
        if generated_lengths
        else 0.0,
    }


def initialize_latent_embedding_from_visible_tokens(model: TinyCausalLM) -> None:
    with torch.no_grad():
        thought_ids = [TOKEN_TO_ID["T"], *(TOKEN_TO_ID[f"X{i}"] for i in range(GRID_SIZE)), *(TOKEN_TO_ID[f"Y{i}"] for i in range(GRID_SIZE))]
        model.token_embedding.weight[LATENT_ID].copy_(model.token_embedding.weight[thought_ids].mean(dim=0))


def split_examples(config: TrainConfig) -> tuple[list[Example], list[Example], list[Example]]:
    train = generate_examples(config.train_size, seed=config.seed)
    used = {example_key(example) for example in train}
    val = generate_examples(config.val_size, seed=config.seed + 1, exclude=used)
    used.update(example_key(example) for example in val)
    test = generate_examples(config.test_size, seed=config.seed + 2, exclude=used)
    return train, val, test


def run_experiment(config: TrainConfig, output_path: Path) -> dict[str, object]:
    global MOVE_COUNT
    MOVE_COUNT = config.move_count
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_examples, val_examples, test_examples = split_examples(config)

    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"train={len(train_examples)} val={len(val_examples)} test={len(test_examples)}", flush=True)

    direct_model = new_model(config, device)
    direct_train = train_model(
        direct_model,
        train_examples,
        val_examples,
        mode="direct",
        config=config,
        steps=config.direct_steps,
        seed=config.seed + 10,
        device=device,
        label="direct_scratch",
    )

    text_model = new_model(config, device)
    text_train = train_model(
        text_model,
        train_examples,
        val_examples,
        mode="visible",
        config=config,
        steps=config.text_steps,
        seed=config.seed + 20,
        device=device,
        label="visible_text",
    )

    latent_from_text = clone_model(text_model).to(device)
    initialize_latent_embedding_from_visible_tokens(latent_from_text)
    latent_train = train_model(
        latent_from_text,
        train_examples,
        val_examples,
        mode="latent",
        config=config,
        steps=config.latent_steps,
        seed=config.seed + 30,
        device=device,
        label="latent_from_text",
    )

    latent_scratch = new_model(config, device)
    initialize_latent_embedding_from_visible_tokens(latent_scratch)
    latent_scratch_train = train_model(
        latent_scratch,
        train_examples,
        val_examples,
        mode="latent",
        config=config,
        steps=config.latent_scratch_steps,
        seed=config.seed + 40,
        device=device,
        label="latent_scratch",
    )

    test_subset = test_examples
    metrics = {
        "direct_scratch": evaluate(direct_model, test_subset, mode="direct", device=device),
        "visible_text": evaluate(text_model, test_subset, mode="visible", device=device),
        "latent_from_text": evaluate(latent_from_text, test_subset, mode="latent", device=device),
        "latent_scratch": evaluate(latent_scratch, test_subset, mode="latent", device=device),
    }

    random_baseline = 1.0 / (GRID_SIZE * GRID_SIZE)
    results = {
        "experiment": "text_training_to_special_latent_token_thought",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "task": {
            "grid_size": GRID_SIZE,
            "move_count": MOVE_COUNT,
            "random_exact_accuracy": random_baseline,
            "prompt_tokens": len(prompt_tokens(test_examples[0])),
            "direct_answer_tokens": 4,
            "visible_reasoning_tokens": MOVE_COUNT * 3 + 4,
            "latent_inserted_tokens": MOVE_COUNT,
            "latent_answer_tokens": 4,
        },
        "config": config.__dict__,
        "train_logs": {
            "direct_scratch": direct_train,
            "visible_text": text_train,
            "latent_from_text": latent_train,
            "latent_scratch": latent_scratch_train,
        },
        "test_metrics": metrics,
        "interpretation": interpret(metrics, random_baseline),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def interpret(metrics: dict[str, dict[str, float | int]], random_baseline: float) -> dict[str, object]:
    latent = float(metrics["latent_from_text"]["exact_accuracy"])
    visible = float(metrics["visible_text"]["exact_accuracy"])
    direct = float(metrics["direct_scratch"]["exact_accuracy"])
    latent_scratch = float(metrics["latent_scratch"]["exact_accuracy"])
    feasible = latent > max(random_baseline * 10.0, 0.25)
    text_transfer_helped = latent >= latent_scratch + 0.05
    near_visible = latent >= max(0.0, visible - 0.15)
    return {
        "mechanism_feasible": feasible,
        "text_to_latent_transfer_helped": text_transfer_helped,
        "latent_near_visible_text": near_visible,
        "latent_beats_direct": latent > direct,
        "summary": (
            "特殊 latent token 路线在该 toy task 上具备初步可行性。"
            if feasible
            else "该配置下未能证明特殊 latent token 路线可行。"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/text_to_latent_thought/results.json")
    parser.add_argument("--quick", action="store_true", help="Run a shorter smoke experiment.")
    parser.add_argument("--move-count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--val-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--text-steps", type=int)
    parser.add_argument("--direct-steps", type=int)
    parser.add_argument("--latent-steps", type=int)
    parser.add_argument("--latent-scratch-steps", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--lr", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig()
    if args.quick:
        config = TrainConfig(
            train_size=1024,
            val_size=256,
            test_size=256,
            batch_size=96,
            text_steps=150,
            direct_steps=150,
            latent_steps=100,
            latent_scratch_steps=100,
            eval_every=75,
            d_model=96,
            n_layers=2,
            n_heads=4,
            lr=8e-4,
        )
    overrides = {
        "move_count": args.move_count,
        "seed": args.seed,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "batch_size": args.batch_size,
        "text_steps": args.text_steps,
        "direct_steps": args.direct_steps,
        "latent_steps": args.latent_steps,
        "latent_scratch_steps": args.latent_scratch_steps,
        "eval_every": args.eval_every,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "lr": args.lr,
    }
    config = replace(config, **{key: value for key, value in overrides.items() if value is not None})
    results = run_experiment(config, Path(args.output))
    print(json.dumps(results["test_metrics"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(results["interpretation"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
