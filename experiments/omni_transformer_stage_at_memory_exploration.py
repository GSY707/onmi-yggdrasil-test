from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from visual_multimodal_stage_ab import write_png  # noqa: E402


COLORS = ("red", "green", "blue", "yellow")
ZONES = 4
TARGETS_PER_EPISODE = 4
POSITIONS = ZONES + 1
HISTORY_LEN = 6

ACTION_SCAN_BASE = 0
ACTION_MOVE_BASE = ACTION_SCAN_BASE + ZONES
ACTION_WRITE_BASE = ACTION_MOVE_BASE + ZONES
ACTION_COLLECT = ACTION_WRITE_BASE + ZONES * len(COLORS)
ACTION_FINAL = ACTION_COLLECT + 1
ACTION_COUNT = ACTION_FINAL + 1

TOKEN_PAD = 0
TOKEN_POS_BASE = 1
TOKEN_TARGET_BASE = TOKEN_POS_BASE + POSITIONS
TOKEN_PHASE_BASE = TOKEN_TARGET_BASE + len(COLORS)
TOKEN_MEMORY_BASE = TOKEN_PHASE_BASE + TARGETS_PER_EPISODE + 1
TOKEN_LAST_SCAN_BASE = TOKEN_MEMORY_BASE + ZONES * (len(COLORS) + 1)
TOKEN_SEEN_BASE = TOKEN_LAST_SCAN_BASE + ZONES * len(COLORS) + 1
TOKEN_LOCAL_BASE = TOKEN_SEEN_BASE + ZONES
TOKEN_HISTORY_BASE = TOKEN_LOCAL_BASE + len(COLORS) + 1
STATE_VOCAB = TOKEN_HISTORY_BASE + ACTION_COUNT

ZONE_COSTS = (3, 4, 5, 6)
SCAN_COST = 3
WRITE_COST = 1
COLLECT_COST = 1
ILLEGAL_COST = 3


@dataclass(frozen=True)
class StageATConfig:
    train_size: int = 3072
    val_size: int = 512
    test_size: int = 512
    batch_size: int = 128
    seed: int = 20260701
    d_model: int = 96
    heads: int = 4
    layers: int = 2
    latent_tokens: int = 6
    lr: float = 8e-4
    steps: int = 700
    eval_every: int = 200
    rollout_budget: int = 45
    sample_count: int = 10


@dataclass(frozen=True)
class EpisodeSpec:
    colors_by_zone: tuple[int, ...]
    targets: tuple[int, ...]
    case_id: str


@dataclass(frozen=True)
class ExploreState:
    spec: EpisodeSpec
    position: int
    phase: int
    memory: tuple[int, ...]
    phase_seen: tuple[int, ...]
    last_scan_zone: int
    last_scan_color: int
    history: tuple[int, ...]
    cost: int
    done: bool
    success: bool


@dataclass(frozen=True)
class MemoryExploreSet:
    states: list[ExploreState]
    tokens: torch.Tensor
    action: torch.Tensor

    def subset(self, indices: list[int]) -> "MemoryExploreSet":
        return MemoryExploreSet(
            states=[self.states[index] for index in indices],
            tokens=self.tokens[indices],
            action=self.action[indices],
        )

    def to(self, device: torch.device) -> "MemoryExploreSet":
        return MemoryExploreSet(self.states, self.tokens.to(device=device), self.action.to(device=device))


def action_name(action: int) -> str:
    if ACTION_SCAN_BASE <= action < ACTION_MOVE_BASE:
        return f"SCAN_ZONE_{action - ACTION_SCAN_BASE}"
    if ACTION_MOVE_BASE <= action < ACTION_WRITE_BASE:
        return f"MOVE_ZONE_{action - ACTION_MOVE_BASE}"
    if ACTION_WRITE_BASE <= action < ACTION_COLLECT:
        offset = action - ACTION_WRITE_BASE
        return f"WRITE_ZONE_{offset // len(COLORS)}_{COLORS[offset % len(COLORS)]}"
    if action == ACTION_COLLECT:
        return "COLLECT_TARGET"
    if action == ACTION_FINAL:
        return "FINAL_REPORT"
    return f"INVALID_{action}"


def initial_state(spec: EpisodeSpec, *, corrupt: bool = False) -> ExploreState:
    memory = [-1] * ZONES
    if corrupt:
        memory = list(spec.colors_by_zone)
        z0 = spec.colors_by_zone.index(spec.targets[0])
        z1 = (z0 + 1) % ZONES
        memory[z0], memory[z1] = memory[z1], memory[z0]
    return ExploreState(
        spec=spec,
        position=0,
        phase=0,
        memory=tuple(memory),
        phase_seen=(0,) * ZONES,
        last_scan_zone=-1,
        last_scan_color=-1,
        history=(),
        cost=0,
        done=False,
        success=False,
    )


def make_episode(index: int, *, rng: random.Random) -> EpisodeSpec:
    colors = list(range(len(COLORS)))
    rng.shuffle(colors)
    targets = colors[:]
    rng.shuffle(targets)
    return EpisodeSpec(tuple(colors), tuple(targets[:TARGETS_PER_EPISODE]), f"case-{index:05d}")


def build_episodes(split: str, size: int, config: StageATConfig) -> list[EpisodeSpec]:
    offsets = {"train": 0, "val": 10_000, "test": 20_000}
    rng = random.Random(config.seed + offsets[split])
    return [make_episode(index, rng=rng) for index in range(size)]


def memory_zone_for(memory: tuple[int, ...], target: int) -> int:
    for zone, color in enumerate(memory):
        if color == target:
            return zone
    return -1


def write_action(zone: int, color: int) -> int:
    return ACTION_WRITE_BASE + zone * len(COLORS) + color


def expert_action(state: ExploreState) -> int:
    if state.phase >= TARGETS_PER_EPISODE:
        return ACTION_FINAL
    target = state.spec.targets[state.phase]
    if state.position > 0:
        zone = state.position - 1
        actual = state.spec.colors_by_zone[zone]
        if actual == target:
            return ACTION_COLLECT
        if state.memory[zone] == target and actual != target:
            if state.last_scan_zone == zone and state.last_scan_color == actual:
                return write_action(zone, actual)
            return ACTION_SCAN_BASE + zone
    if state.last_scan_zone >= 0 and state.memory[state.last_scan_zone] != state.last_scan_color:
        return write_action(state.last_scan_zone, state.last_scan_color)
    if state.phase == 0:
        for zone, color in enumerate(state.memory):
            if color < 0:
                return ACTION_SCAN_BASE + zone
    known_zone = memory_zone_for(state.memory, target)
    if known_zone >= 0:
        return ACTION_MOVE_BASE + known_zone
    for zone, seen in enumerate(state.phase_seen):
        if not seen:
            return ACTION_SCAN_BASE + zone
    return ACTION_FINAL


def memoryless_oracle_action(state: ExploreState) -> int:
    if state.phase >= TARGETS_PER_EPISODE:
        return ACTION_FINAL
    target = state.spec.targets[state.phase]
    if state.position > 0 and state.spec.colors_by_zone[state.position - 1] == target:
        return ACTION_COLLECT
    if state.last_scan_zone >= 0 and state.last_scan_color == target:
        return ACTION_MOVE_BASE + state.last_scan_zone
    for zone, seen in enumerate(state.phase_seen):
        if not seen:
            return ACTION_SCAN_BASE + zone
    return ACTION_FINAL


def append_history(history: tuple[int, ...], action: int) -> tuple[int, ...]:
    return (history + (action,))[-HISTORY_LEN:]


def step_state(state: ExploreState, action: int, *, forget_memory: bool = False) -> ExploreState:
    if state.done:
        return state
    memory = list(state.memory)
    phase_seen = list(state.phase_seen)
    position = state.position
    phase = state.phase
    last_scan_zone = state.last_scan_zone
    last_scan_color = state.last_scan_color
    cost = state.cost
    done = False
    success = False
    legal = True
    if ACTION_SCAN_BASE <= action < ACTION_MOVE_BASE:
        zone = action - ACTION_SCAN_BASE
        phase_seen[zone] = 1
        last_scan_zone = zone
        last_scan_color = state.spec.colors_by_zone[zone]
        cost += SCAN_COST
    elif ACTION_MOVE_BASE <= action < ACTION_WRITE_BASE:
        zone = action - ACTION_MOVE_BASE
        position = zone + 1
        cost += ZONE_COSTS[zone]
    elif ACTION_WRITE_BASE <= action < ACTION_COLLECT:
        offset = action - ACTION_WRITE_BASE
        zone = offset // len(COLORS)
        color = offset % len(COLORS)
        if last_scan_zone == zone and last_scan_color == color:
            memory[zone] = color
            cost += WRITE_COST
        else:
            legal = False
    elif action == ACTION_COLLECT:
        if phase < TARGETS_PER_EPISODE and position > 0 and state.spec.colors_by_zone[position - 1] == state.spec.targets[phase]:
            phase += 1
            position = 0
            phase_seen = [0] * ZONES
            last_scan_zone = -1
            last_scan_color = -1
            cost += COLLECT_COST
        else:
            legal = False
    elif action == ACTION_FINAL:
        done = True
        success = phase >= TARGETS_PER_EPISODE
    else:
        legal = False
    if not legal:
        cost += ILLEGAL_COST
    if forget_memory:
        memory = [-1] * ZONES
    return ExploreState(
        spec=state.spec,
        position=position,
        phase=phase,
        memory=tuple(memory),
        phase_seen=tuple(phase_seen),
        last_scan_zone=last_scan_zone,
        last_scan_color=last_scan_color,
        history=append_history(state.history, action),
        cost=cost,
        done=done,
        success=success,
    )


def state_tokens(state: ExploreState, *, no_memory: bool = False) -> list[int]:
    memory = (-1,) * ZONES if no_memory else state.memory
    target = 0 if state.phase >= TARGETS_PER_EPISODE else state.spec.targets[state.phase]
    tokens = [
        TOKEN_POS_BASE + state.position,
        TOKEN_TARGET_BASE + target,
        TOKEN_PHASE_BASE + state.phase,
    ]
    for zone, color in enumerate(memory):
        color_index = color + 1
        tokens.append(TOKEN_MEMORY_BASE + zone * (len(COLORS) + 1) + color_index)
    if state.last_scan_zone >= 0:
        tokens.append(TOKEN_LAST_SCAN_BASE + state.last_scan_zone * len(COLORS) + state.last_scan_color)
    else:
        tokens.append(TOKEN_LAST_SCAN_BASE + ZONES * len(COLORS))
    for zone, seen in enumerate(state.phase_seen):
        tokens.append(TOKEN_SEEN_BASE + zone if seen else TOKEN_PAD)
    local_color = state.spec.colors_by_zone[state.position - 1] if state.position > 0 else len(COLORS)
    tokens.append(TOKEN_LOCAL_BASE + local_color)
    padded_history = state.history[-HISTORY_LEN:] + (ACTION_FINAL,) * (HISTORY_LEN - len(state.history[-HISTORY_LEN:]))
    tokens.extend(TOKEN_HISTORY_BASE + action for action in padded_history)
    return tokens


def collect_supervised_states(episodes: list[EpisodeSpec], *, include_corrupt: bool) -> MemoryExploreSet:
    states: list[ExploreState] = []
    actions: list[int] = []
    for spec in episodes:
        variants = [initial_state(spec)]
        if include_corrupt:
            variants.append(initial_state(spec, corrupt=True))
        for start in variants:
            state = start
            for _ in range(40):
                action = expert_action(state)
                states.append(state)
                actions.append(action)
                state = step_state(state, action)
                if state.done:
                    break
    return MemoryExploreSet(
        states=states,
        tokens=torch.tensor([state_tokens(state) for state in states], dtype=torch.long),
        action=torch.tensor(actions, dtype=torch.long),
    )


def random_batch(data: MemoryExploreSet, *, rng: random.Random, batch_size: int, device: torch.device) -> MemoryExploreSet:
    return data.subset([rng.randrange(len(data.states)) for _ in range(batch_size)]).to(device)


class MemoryExploreModel(nn.Module):
    def __init__(self, config: StageATConfig, seq_len: int) -> None:
        super().__init__()
        self.token = nn.Embedding(STATE_VOCAB, config.d_model, padding_idx=TOKEN_PAD)
        self.position = nn.Parameter(torch.randn(seq_len + config.latent_tokens, config.d_model) * 0.02)
        self.latent = nn.Parameter(torch.randn(config.latent_tokens, config.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.action = nn.Linear(config.d_model, ACTION_COUNT)

    def forward(self, tokens: torch.Tensor, *, no_memory: bool = False) -> torch.Tensor:
        if no_memory:
            tokens = tokens.clone()
            mem_start = 3
            tokens[:, mem_start : mem_start + ZONES] = torch.tensor(
                [TOKEN_MEMORY_BASE + zone * (len(COLORS) + 1) for zone in range(ZONES)],
                dtype=torch.long,
                device=tokens.device,
            ).view(1, -1)
        x = self.token(tokens)
        latent = self.latent.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        x = torch.cat((x, latent), dim=1) + self.position[: x.shape[1] + latent.shape[1]].view(1, -1, x.shape[-1])
        encoded = self.encoder(x)
        pooled = self.norm(encoded[:, -latent.shape[1] :]).mean(dim=1)
        return self.action(pooled)


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == target).float().mean().item())


def train_model(model: MemoryExploreModel, train: MemoryExploreSet, val: MemoryExploreSet, config: StageATConfig, device: torch.device) -> dict[str, object]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    rng = random.Random(config.seed + 313)
    history = []
    for step in range(1, config.steps + 1):
        model.train()
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        logits = model(batch.tokens)
        loss = F.cross_entropy(logits, batch.action)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            metrics = evaluate_action(model, val, device, prefix="val")
            metrics["step"] = float(step)
            metrics["loss"] = float(loss.detach().item())
            history.append(metrics)
            print(json.dumps(metrics), flush=True)
    return {"history": history}


@torch.no_grad()
def evaluate_action(model: MemoryExploreModel, data: MemoryExploreSet, device: torch.device, *, prefix: str) -> dict[str, float]:
    model.eval()
    batch = data.to(device)
    logits = model(batch.tokens)
    no_memory = model(batch.tokens, no_memory=True)
    return {
        f"{prefix}_action_accuracy": accuracy(logits, batch.action),
        f"{prefix}_no_memory_action_accuracy": accuracy(no_memory, batch.action),
    }


@torch.no_grad()
def rollout_episode(
    model: MemoryExploreModel,
    spec: EpisodeSpec,
    device: torch.device,
    *,
    budget: int,
    no_memory: bool = False,
    corrupt: bool = False,
    oracle_no_memory: bool = False,
    capture_trace: bool = False,
) -> dict[str, object]:
    model.eval()
    state = initial_state(spec, corrupt=corrupt)
    trace = []
    scans = 0
    corrections = 0
    for _ in range(50):
        if oracle_no_memory:
            action = memoryless_oracle_action(state)
        else:
            tokens = torch.tensor([state_tokens(state, no_memory=no_memory)], dtype=torch.long, device=device)
            action = int(model(tokens, no_memory=no_memory).argmax(dim=-1).item())
        before = state
        if action_name(action).startswith("SCAN_ZONE"):
            scans += 1
        state = step_state(state, action, forget_memory=no_memory)
        if before.position > 0:
            zone = before.position - 1
            if before.memory[zone] >= 0 and before.memory[zone] != before.spec.colors_by_zone[zone] and state.memory[zone] == before.spec.colors_by_zone[zone]:
                corrections += 1
        if capture_trace:
            trace.append(
                {
                    "action": action_name(action),
                    "before": {
                        "position": before.position,
                        "phase": before.phase,
                        "target": COLORS[before.spec.targets[min(before.phase, TARGETS_PER_EPISODE - 1)]],
                        "memory": [None if value < 0 else COLORS[value] for value in before.memory],
                        "last_scan": None
                        if before.last_scan_zone < 0
                        else {"zone": before.last_scan_zone, "color": COLORS[before.last_scan_color]},
                        "cost": before.cost,
                    },
                    "after_cost": state.cost,
                }
            )
        if state.done or state.cost > budget:
            break
    return {
        "success": bool(state.success and state.cost <= budget),
        "raw_success": bool(state.success),
        "cost": state.cost,
        "scans": scans,
        "corrections": corrections,
        "steps": len(state.history),
        "trace": trace,
    }


def evaluate_rollouts(model: MemoryExploreModel, episodes: list[EpisodeSpec], device: torch.device, config: StageATConfig) -> dict[str, float]:
    full = [rollout_episode(model, spec, device, budget=config.rollout_budget) for spec in episodes]
    no_memory = [rollout_episode(model, spec, device, budget=config.rollout_budget, no_memory=True, oracle_no_memory=True) for spec in episodes]
    no_memory_model = [rollout_episode(model, spec, device, budget=config.rollout_budget, no_memory=True) for spec in episodes]
    corrupt = [rollout_episode(model, spec, device, budget=config.rollout_budget + 8, corrupt=True) for spec in episodes]
    return {
        "test_episode_success": statistics.mean(1.0 if row["success"] else 0.0 for row in full),
        "test_mean_cost": statistics.mean(float(row["cost"]) for row in full),
        "test_mean_scans": statistics.mean(float(row["scans"]) for row in full),
        "test_no_memory_episode_success": statistics.mean(1.0 if row["success"] else 0.0 for row in no_memory),
        "test_no_memory_mean_cost": statistics.mean(float(row["cost"]) for row in no_memory),
        "test_no_memory_mean_scans": statistics.mean(float(row["scans"]) for row in no_memory),
        "test_no_memory_model_episode_success": statistics.mean(1.0 if row["success"] else 0.0 for row in no_memory_model),
        "test_no_memory_model_mean_cost": statistics.mean(float(row["cost"]) for row in no_memory_model),
        "test_corrupt_memory_success": statistics.mean(1.0 if row["success"] else 0.0 for row in corrupt),
        "test_corrupt_memory_correction_rate": statistics.mean(1.0 if row["corrections"] > 0 else 0.0 for row in corrupt),
        "test_corrupt_memory_mean_cost": statistics.mean(float(row["cost"]) for row in corrupt),
    }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def render_episode_map(spec: EpisodeSpec) -> torch.Tensor:
    image = torch.full((3, 96, 96), 0.08)
    image[:, 40:56, :] = 0.18
    image[:, :, 40:56] = 0.18
    palette = torch.tensor(
        [
            [0.88, 0.10, 0.10],
            [0.10, 0.70, 0.25],
            [0.15, 0.32, 0.90],
            [0.94, 0.74, 0.12],
        ]
    )
    boxes = ((8, 8), (64, 8), (8, 64), (64, 64))
    for zone, (x, y) in enumerate(boxes):
        image[:, y : y + 24, x : x + 24] = palette[spec.colors_by_zone[zone]].view(3, 1, 1)
        image[:, y + 10 : y + 14, 44:52] = 0.65
        image[:, 44:52, x + 10 : x + 14] = 0.65
    image[:, 42:54, 42:54] = 0.82
    return image


def write_samples(model: MemoryExploreModel, episodes: list[EpisodeSpec], path: Path, device: torch.device, config: StageATConfig) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in episodes[: config.sample_count]:
        png = path / f"{spec.case_id}_map.png"
        write_png(png, render_episode_map(spec))
        rows.append(
            {
                "case_id": spec.case_id,
                "map_png": str(png),
                "zone_colors": [COLORS[color] for color in spec.colors_by_zone],
                "targets": [COLORS[color] for color in spec.targets],
                "memory_trace": rollout_episode(model, spec, device, budget=config.rollout_budget, capture_trace=True),
                "no_memory_trace": rollout_episode(
                    model,
                    spec,
                    device,
                    budget=config.rollout_budget,
                    no_memory=True,
                    oracle_no_memory=True,
                    capture_trace=True,
                ),
                "no_memory_model_trace": rollout_episode(model, spec, device, budget=config.rollout_budget, no_memory=True, capture_trace=True),
                "corrupt_trace": rollout_episode(model, spec, device, budget=config.rollout_budget + 8, corrupt=True, capture_trace=True),
            }
        )
    (path / "sample_traces.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def run_one(config: StageATConfig, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_episodes = build_episodes("train", config.train_size, config)
    val_episodes = build_episodes("val", config.val_size, config)
    test_episodes = build_episodes("test", config.test_size, config)
    train = collect_supervised_states(train_episodes, include_corrupt=True)
    val = collect_supervised_states(val_episodes, include_corrupt=True)
    test = collect_supervised_states(test_episodes, include_corrupt=True)
    model = MemoryExploreModel(config, train.tokens.shape[1])
    training = train_model(model, train, val, config, device)
    action_metrics = evaluate_action(model, test, device, prefix="test")
    rollout_metrics = evaluate_rollouts(model, test_episodes, device, config)
    write_samples(model, test_episodes, output_path.parent / "sample_traces", device, config)
    result = {
        "schema_version": 1,
        "stage": "AT",
        "config": asdict(config),
        "device": str(device),
        "metrics": {**action_metrics, **rollout_metrics},
        "cost": {"elapsed_sec": time.perf_counter() - started, "parameters": parameter_count(model)},
        "training": training,
        "interpretation": {
            "goal": "Measure whether local memory plus active reads lowers multi-target exploration cost under a fixed budget.",
            "success_condition": "Memory rollout success must exceed an oracle no-memory policy by at least 30pp, with lower cost and corrupt-memory correction.",
            "boundary": "Controlled abstract grid with scan tools and color beacons; not a real embodied navigation benchmark.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "metrics": result["metrics"]}, ensure_ascii=False, indent=2), flush=True)
    return result


def collect_numbers(value: object, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}_{key}" if prefix else str(key)
            out.update(collect_numbers(child, name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)
    return out


def aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    rows = [collect_numbers(run) for run in runs]
    keys = sorted(set().union(*(row.keys() for row in rows)))
    summary = {key: statistics.mean(row[key] for row in rows if key in row) for key in keys}
    stdev = {
        key: statistics.pstdev([row[key] for row in rows if key in row])
        for key in keys
        if sum(1 for row in rows if key in row) > 1
    }
    success_gap = summary["metrics_test_episode_success"] - summary["metrics_test_no_memory_episode_success"]
    scan_reduction = summary["metrics_test_no_memory_mean_scans"] - summary["metrics_test_mean_scans"]
    cost_reduction = summary["metrics_test_no_memory_mean_cost"] - summary["metrics_test_mean_cost"]
    return {
        "schema_version": 1,
        "stage": "AT",
        "runs": runs,
        "summary": summary,
        "stdev": stdev,
        "gates": {
            "episode_success": summary["metrics_test_episode_success"],
            "no_memory_episode_success": summary["metrics_test_no_memory_episode_success"],
            "success_gap": success_gap,
            "scan_reduction": scan_reduction,
            "cost_reduction": cost_reduction,
            "corrupt_memory_success": summary["metrics_test_corrupt_memory_success"],
            "corrupt_memory_correction_rate": summary["metrics_test_corrupt_memory_correction_rate"],
            "passes_memory_gap_30pp": success_gap >= 0.30,
            "passes_lower_scan_and_cost": scan_reduction > 0 and cost_reduction > 0,
            "passes_corrupt_correction_85": summary["metrics_test_corrupt_memory_correction_rate"] >= 0.85,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_at_memory_exploration/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_at_memory_exploration/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_at_memory_exploration/sweep_results.json"))
    parser.add_argument("--train-size", type=int, default=StageATConfig.train_size)
    parser.add_argument("--val-size", type=int, default=StageATConfig.val_size)
    parser.add_argument("--test-size", type=int, default=StageATConfig.test_size)
    parser.add_argument("--batch-size", type=int, default=StageATConfig.batch_size)
    parser.add_argument("--d-model", type=int, default=StageATConfig.d_model)
    parser.add_argument("--heads", type=int, default=StageATConfig.heads)
    parser.add_argument("--layers", type=int, default=StageATConfig.layers)
    parser.add_argument("--latent-tokens", type=int, default=StageATConfig.latent_tokens)
    parser.add_argument("--steps", type=int, default=StageATConfig.steps)
    parser.add_argument("--eval-every", type=int, default=StageATConfig.eval_every)
    parser.add_argument("--rollout-budget", type=int, default=StageATConfig.rollout_budget)
    parser.add_argument("--sample-count", type=int, default=StageATConfig.sample_count)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace, seed: int) -> StageATConfig:
    return StageATConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        seed=seed,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        latent_tokens=args.latent_tokens,
        steps=args.steps,
        eval_every=args.eval_every,
        rollout_budget=args.rollout_budget,
        sample_count=args.sample_count,
    )


def main() -> None:
    args = parse_args()
    if args.sweep:
        runs = []
        for seed_text in args.seeds.split(","):
            seed = int(seed_text.strip())
            runs.append(run_one(config_from_args(args, seed), args.output_dir / f"seed{seed}.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"aggregate": aggregate["gates"]}, ensure_ascii=False, indent=2), flush=True)
    else:
        first_seed = int(args.seeds.split(",")[0].strip())
        run_one(config_from_args(args, first_seed), args.output)


if __name__ == "__main__":
    main()
