from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
import time
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from multimodal_fusion_latent_flow import IMAGE_SIZE, TELEMETRY_CHANNELS, render_telemetry, stat
from omni_transformer_stage_h import PATCH_COUNT, PATCH_SIZE, TransformerBlock
from omni_transformer_stage_h5 import render_augmented_panel_images
from visual_multimodal_stage_ab import write_png


FAULTS = 4
EPISODE_TYPES = 4
TEXT_LEN = 8
MEMORY_LEN = 3
HISTORY_LEN = 12
TELEMETRY_STEPS = 12
LATENT_TOKENS = 8

EP_DIAGNOSE = 0
EP_MISSING_INFO = 1
EP_CONFLICTING_SIGNALS = 2
EP_MEMORY_AIDED = 3
EPISODE_NAMES = {
    EP_DIAGNOSE: "diagnose_and_fix",
    EP_MISSING_INFO: "missing_info",
    EP_CONFLICTING_SIGNALS: "conflicting_signals",
    EP_MEMORY_AIDED: "memory_aided",
}

ACTION_QUERY_LOG = 0
ACTION_RUN_TEST = 1
ACTION_APPLY_BASE = 2
ACTION_WRITE_BASE = ACTION_APPLY_BASE + FAULTS
ACTION_FINAL_BASE = ACTION_WRITE_BASE + FAULTS
ACTION_VOCAB = ACTION_FINAL_BASE + FAULTS

ACTION_NAMES = {
    ACTION_QUERY_LOG: "TOOL_QUERY_LOG",
    ACTION_RUN_TEST: "TOOL_RUN_TEST",
    **{ACTION_APPLY_BASE + i: f"TOOL_APPLY_FIX_{i}" for i in range(FAULTS)},
    **{ACTION_WRITE_BASE + i: f"MEMORY_WRITE_{i}" for i in range(FAULTS)},
    **{ACTION_FINAL_BASE + i: f"FINAL_REPORT_{i}" for i in range(FAULTS)},
}

HIST_PAD = 0
HIST_ACTION_BASE = 1
HIST_LOG_BASE = HIST_ACTION_BASE + ACTION_VOCAB
HIST_TEST_BASE = HIST_LOG_BASE + FAULTS
HIST_APPLY_OK_BASE = HIST_TEST_BASE + FAULTS
HIST_WRITE_OK_BASE = HIST_APPLY_OK_BASE + FAULTS
HIST_FINAL_BASE = HIST_WRITE_OK_BASE + FAULTS
HISTORY_VOCAB = HIST_FINAL_BASE + FAULTS

TEXT_VOCAB = 32
MEMORY_VOCAB = 8
SPECIAL_BOS = 0
SPECIAL_IMG = 1
SPECIAL_TEXT = 2
SPECIAL_TELEMETRY = 3
SPECIAL_MEMORY = 4
SPECIAL_HISTORY = 5
SPECIAL_ACTION = 6
SPECIAL_COUNT = 7


@dataclass(frozen=True)
class AgentConfig:
    train_size: int = 4096
    val_size: int = 1024
    test_size: int = 1024
    batch_size: int = 128
    seed: int = 20260701
    d_model: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.0
    lr: float = 8e-4
    direct_steps: int = 750
    bottleneck_steps: int = 850
    probe_steps: int = 200
    eval_every: int = 425
    max_rollout_steps: int = 5


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: int
    episode_type: int
    fault: int
    visual_cue: int
    telemetry_cue: int
    memory_fault: int


@dataclass(frozen=True)
class AgentState:
    spec: EpisodeSpec
    history: tuple[int, ...]
    fixed_fault: int
    memory_written: int
    done: bool
    step: int


@dataclass
class AgentBatch:
    image: torch.Tensor
    text: torch.Tensor
    telemetry: torch.Tensor
    memory: torch.Tensor
    history: torch.Tensor
    action: torch.Tensor
    fault: torch.Tensor
    phase: torch.Tensor
    episode_type: torch.Tensor


@dataclass(frozen=True)
class AgentLayout:
    input_end: int
    latent_start: int
    latent_end: int
    action_pos: int
    seq_len: int


def action_to_text(action: int) -> str:
    return ACTION_NAMES[int(action)]


def history_has(history: tuple[int, ...], start: int, count: int) -> int:
    for token in history:
        if start <= token < start + count:
            return token - start
    return -1


def has_log(history: tuple[int, ...]) -> int:
    return history_has(history, HIST_LOG_BASE, FAULTS)


def has_test(history: tuple[int, ...]) -> int:
    return history_has(history, HIST_TEST_BASE, FAULTS)


def history_tokens(history: tuple[int, ...]) -> tuple[int, ...]:
    trimmed = history[-HISTORY_LEN:]
    return trimmed + (HIST_PAD,) * (HISTORY_LEN - len(trimmed))


def make_episode(index: int, *, rng: random.Random) -> EpisodeSpec:
    episode_type = index % EPISODE_TYPES
    fault = rng.randrange(FAULTS)
    if episode_type == EP_DIAGNOSE:
        visual_cue = fault
        telemetry_cue = fault
        memory_fault = -1
    elif episode_type == EP_MISSING_INFO:
        visual_cue = (fault + 1) % FAULTS
        telemetry_cue = (fault + 2) % FAULTS
        memory_fault = -1
    elif episode_type == EP_CONFLICTING_SIGNALS:
        visual_cue = (fault + 1) % FAULTS
        telemetry_cue = (fault + 2) % FAULTS
        memory_fault = -1
    else:
        visual_cue = (fault + 1) % FAULTS
        telemetry_cue = (fault + 2) % FAULTS
        memory_fault = fault
    return EpisodeSpec(index, episode_type, fault, visual_cue, telemetry_cue, memory_fault)


def generate_episodes(size: int, *, seed: int) -> list[EpisodeSpec]:
    rng = random.Random(seed)
    episodes = [make_episode(index, rng=rng) for index in range(size)]
    rng.shuffle(episodes)
    return episodes


def phase_for(state: AgentState) -> int:
    if state.fixed_fault < 0:
        if state.spec.episode_type == EP_MISSING_INFO and has_log(state.history) < 0:
            return 0
        if state.spec.episode_type == EP_CONFLICTING_SIGNALS and has_test(state.history) < 0:
            return 1
        return 2
    if state.memory_written < 0:
        return 3
    return 4


def expert_action(state: AgentState) -> int:
    fault = state.spec.fault
    if state.fixed_fault < 0:
        if state.spec.episode_type == EP_MISSING_INFO and has_log(state.history) < 0:
            return ACTION_QUERY_LOG
        if state.spec.episode_type == EP_CONFLICTING_SIGNALS and has_test(state.history) < 0:
            return ACTION_RUN_TEST
        return ACTION_APPLY_BASE + fault
    if state.memory_written < 0:
        return ACTION_WRITE_BASE + fault
    return ACTION_FINAL_BASE + fault


def step_environment(state: AgentState, action: int) -> AgentState:
    history = list(state.history)
    fixed_fault = state.fixed_fault
    memory_written = state.memory_written
    done = state.done
    fault = state.spec.fault
    history.append(HIST_ACTION_BASE + action)
    if action == ACTION_QUERY_LOG:
        history.append(HIST_LOG_BASE + fault)
    elif action == ACTION_RUN_TEST:
        history.append(HIST_TEST_BASE + fault)
    elif ACTION_APPLY_BASE <= action < ACTION_APPLY_BASE + FAULTS:
        applied = action - ACTION_APPLY_BASE
        if applied == fault:
            fixed_fault = fault
        history.append(HIST_APPLY_OK_BASE + applied)
    elif ACTION_WRITE_BASE <= action < ACTION_WRITE_BASE + FAULTS:
        written = action - ACTION_WRITE_BASE
        if written == fault:
            memory_written = fault
        history.append(HIST_WRITE_OK_BASE + written)
    elif ACTION_FINAL_BASE <= action < ACTION_FINAL_BASE + FAULTS:
        done = True
        history.append(HIST_FINAL_BASE + (action - ACTION_FINAL_BASE))
    return AgentState(
        spec=state.spec,
        history=tuple(history_tokens(tuple(history))),
        fixed_fault=fixed_fault,
        memory_written=memory_written,
        done=done,
        step=state.step + 1,
    )


def initial_state(spec: EpisodeSpec) -> AgentState:
    return AgentState(spec=spec, history=(), fixed_fault=-1, memory_written=-1, done=False, step=0)


def unroll_expert(episodes: list[EpisodeSpec]) -> list[AgentState]:
    states: list[AgentState] = []
    for spec in episodes:
        state = initial_state(spec)
        for _ in range(5):
            states.append(state)
            action = expert_action(state)
            state = step_environment(state, action)
            if state.done:
                break
    return states


def render_text_tokens(states: list[AgentState], *, device: torch.device) -> torch.Tensor:
    tokens = torch.zeros((len(states), TEXT_LEN), dtype=torch.long, device=device)
    for idx, state in enumerate(states):
        spec = state.spec
        tokens[idx, 0] = 1
        tokens[idx, 1] = 3 + spec.episode_type
        tokens[idx, 2] = 8 + spec.visual_cue
        tokens[idx, 3] = 12 + spec.telemetry_cue
        tokens[idx, 4] = 16 + phase_for(state)
        tokens[idx, 5] = 21 + min(state.step, 5)
        tokens[idx, 6] = 27
        tokens[idx, 7] = 2
    return tokens


def render_memory_tokens(states: list[AgentState], *, device: torch.device) -> torch.Tensor:
    tokens = torch.zeros((len(states), MEMORY_LEN), dtype=torch.long, device=device)
    for idx, state in enumerate(states):
        if state.spec.memory_fault >= 0:
            tokens[idx, 0] = 1 + state.spec.memory_fault
        if state.memory_written >= 0:
            tokens[idx, 1] = 1 + state.memory_written
        tokens[idx, 2] = 5 + state.spec.episode_type % 2
    return tokens


def make_batch(states: list[AgentState], *, device: torch.device) -> AgentBatch:
    visual = torch.tensor([state.spec.visual_cue for state in states], dtype=torch.long, device=device)
    telemetry_cue = torch.tensor([state.spec.telemetry_cue for state in states], dtype=torch.long, device=device)
    variant = torch.tensor([(state.spec.episode_id + state.step) % 7 for state in states], dtype=torch.long, device=device)
    return AgentBatch(
        image=render_augmented_panel_images(visual, device=device, variant_ids=variant),
        text=render_text_tokens(states, device=device),
        telemetry=render_telemetry(telemetry_cue, device=device),
        memory=render_memory_tokens(states, device=device),
        history=torch.tensor([history_tokens(state.history) for state in states], dtype=torch.long, device=device),
        action=torch.tensor([expert_action(state) for state in states], dtype=torch.long, device=device),
        fault=torch.tensor([state.spec.fault for state in states], dtype=torch.long, device=device),
        phase=torch.tensor([phase_for(state) for state in states], dtype=torch.long, device=device),
        episode_type=torch.tensor([state.spec.episode_type for state in states], dtype=torch.long, device=device),
    )


def random_batch(states: list[AgentState], *, rng: random.Random, batch_size: int, device: torch.device) -> AgentBatch:
    return make_batch(rng.choices(states, k=batch_size), device=device)


class AgentTransformer(nn.Module):
    def __init__(self, config: AgentConfig) -> None:
        super().__init__()
        self.config = config
        self.image_patch = nn.Linear(3 * PATCH_SIZE * PATCH_SIZE, config.d_model)
        self.telemetry_projection = nn.Linear(TELEMETRY_CHANNELS, config.d_model)
        self.text_embedding = nn.Embedding(TEXT_VOCAB, config.d_model)
        self.memory_embedding = nn.Embedding(MEMORY_VOCAB, config.d_model)
        self.history_embedding = nn.Embedding(HISTORY_VOCAB, config.d_model)
        self.special_embedding = nn.Embedding(SPECIAL_COUNT, config.d_model)
        self.latent_embedding = nn.Parameter(torch.randn(LATENT_TOKENS, config.d_model) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(128, config.d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config.d_model, config.heads, config.dropout) for _ in range(config.layers)]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.action_head = nn.Linear(config.d_model, ACTION_VOCAB)

    def layout(self) -> AgentLayout:
        input_end = 1 + 1 + PATCH_COUNT + 1 + TEXT_LEN + 1 + TELEMETRY_STEPS + 1 + MEMORY_LEN + 1 + HISTORY_LEN
        latent_start = input_end
        latent_end = latent_start + LATENT_TOKENS
        action_pos = latent_end
        return AgentLayout(input_end=input_end, latent_start=latent_start, latent_end=latent_end, action_pos=action_pos, seq_len=action_pos + 1)

    def patch_image(self, image: torch.Tensor) -> torch.Tensor:
        patches = image.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches.view(image.shape[0], PATCH_COUNT, 3 * PATCH_SIZE * PATCH_SIZE)

    def build_embeddings(
        self,
        batch: AgentBatch,
        *,
        zero_modalities: Iterable[str] = (),
    ) -> tuple[torch.Tensor, AgentLayout]:
        zero_set = set(zero_modalities)
        batch_size = int(batch.action.shape[0])
        special = self.special_embedding.weight
        image_tokens = self.image_patch(self.patch_image(batch.image))
        text_tokens = self.text_embedding(batch.text)
        telemetry_tokens = self.telemetry_projection(batch.telemetry)
        memory_tokens = self.memory_embedding(batch.memory)
        history_tokens = self.history_embedding(batch.history)
        if "image" in zero_set:
            image_tokens = torch.zeros_like(image_tokens)
        if "text" in zero_set:
            text_tokens = torch.zeros_like(text_tokens)
        if "telemetry" in zero_set:
            telemetry_tokens = torch.zeros_like(telemetry_tokens)
        if "memory" in zero_set:
            memory_tokens = torch.zeros_like(memory_tokens)
        if "history" in zero_set:
            history_tokens = torch.zeros_like(history_tokens)

        chunks = [
            special[SPECIAL_BOS].expand(batch_size, 1, -1),
            special[SPECIAL_IMG].expand(batch_size, 1, -1),
            image_tokens,
            special[SPECIAL_TEXT].expand(batch_size, 1, -1),
            text_tokens,
            special[SPECIAL_TELEMETRY].expand(batch_size, 1, -1),
            telemetry_tokens,
            special[SPECIAL_MEMORY].expand(batch_size, 1, -1),
            memory_tokens,
            special[SPECIAL_HISTORY].expand(batch_size, 1, -1),
            history_tokens,
            self.latent_embedding.expand(batch_size, LATENT_TOKENS, -1),
            special[SPECIAL_ACTION].expand(batch_size, 1, -1),
        ]
        x = torch.cat(chunks, dim=1)
        x = x + self.position_embedding[: x.shape[1]].unsqueeze(0)
        return x, self.layout()

    def attention_mask(self, layout: AgentLayout, *, mode: str, disable_latent_to_action: bool, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(layout.seq_len, layout.seq_len, dtype=torch.bool, device=device), diagonal=1)
        if mode == "latent_bottleneck":
            mask[layout.action_pos, : layout.input_end] = True
            if disable_latent_to_action:
                mask[layout.action_pos, layout.latent_start : layout.latent_end] = True
        elif mode != "direct":
            raise ValueError(f"unknown mode: {mode}")
        return mask

    def forward(
        self,
        batch: AgentBatch,
        *,
        mode: str,
        zero_modalities: Iterable[str] = (),
        disable_latent_to_action: bool = False,
        return_hidden: bool = False,
    ) -> dict[str, torch.Tensor | AgentLayout]:
        x, layout = self.build_embeddings(batch, zero_modalities=zero_modalities)
        mask = self.attention_mask(layout, mode=mode, disable_latent_to_action=disable_latent_to_action, device=x.device)
        for block in self.blocks:
            x = block(x, mask)
        hidden = self.norm(x)
        output: dict[str, torch.Tensor | AgentLayout] = {"logits": self.action_head(hidden[:, layout.action_pos]), "layout": layout}
        if return_hidden:
            output["hidden"] = hidden
        return output


@torch.no_grad()
def evaluate_steps(
    model: AgentTransformer,
    states: list[AgentState],
    config: AgentConfig,
    *,
    device: torch.device,
    mode: str,
    zero_modalities: Iterable[str] = (),
    disable_latent_to_action: bool = False,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    for start in range(0, len(states), config.batch_size):
        batch = make_batch(states[start : start + config.batch_size], device=device)
        output = model(
            batch,
            mode=mode,
            zero_modalities=zero_modalities,
            disable_latent_to_action=disable_latent_to_action,
        )
        predicted = output["logits"].argmax(dim=-1)
        correct += int((predicted == batch.action).sum().detach().cpu())
        total += len(batch.action)
    model.train()
    return {"step_action_exact": correct / total}


def train_model(
    model: AgentTransformer,
    train_states: list[AgentState],
    val_states: list[AgentState],
    config: AgentConfig,
    *,
    steps: int,
    seed: int,
    device: torch.device,
    mode: str,
    label: str,
) -> dict[str, object]:
    rng = random.Random(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        batch = random_batch(train_states, rng=rng, batch_size=config.batch_size, device=device)
        output = model(batch, mode=mode)
        loss = F.cross_entropy(output["logits"], batch.action)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate_steps(model, val_states, config, device=device, mode=mode)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), **metrics})
            print(f"{label} step={step:4d} loss={float(loss.detach().cpu()):.4f} step_exact={metrics['step_action_exact']:.3f}", flush=True)
    return {
        "history": history,
        "training_seconds": round(time.perf_counter() - started, 3),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


@torch.no_grad()
def rollout(
    model: AgentTransformer,
    episodes: list[EpisodeSpec],
    config: AgentConfig,
    *,
    device: torch.device,
    mode: str,
    zero_modalities: Iterable[str] = (),
    disable_latent_to_action: bool = False,
) -> dict[str, object]:
    model.eval()
    states = [initial_state(spec) for spec in episodes]
    finished = [False for _ in episodes]
    invalid_actions = 0
    total_actions = 0
    extra_tools = 0
    for _ in range(config.max_rollout_steps):
        active_indices = [idx for idx, done in enumerate(finished) if not done]
        if not active_indices:
            break
        active_states = [states[idx] for idx in active_indices]
        batch = make_batch(active_states, device=device)
        output = model(
            batch,
            mode=mode,
            zero_modalities=zero_modalities,
            disable_latent_to_action=disable_latent_to_action,
        )
        actions = output["logits"].argmax(dim=-1).detach().cpu().tolist()
        for idx, action in zip(active_indices, actions):
            total_actions += 1
            expected = expert_action(states[idx])
            if action != expected:
                invalid_actions += 1
            if action in (ACTION_QUERY_LOG, ACTION_RUN_TEST) and action != expected:
                extra_tools += 1
            states[idx] = step_environment(states[idx], int(action))
            finished[idx] = states[idx].done

    success_by_type: dict[str, list[float]] = {name: [] for name in EPISODE_NAMES.values()}
    tool_counts: list[int] = []
    successes: list[float] = []
    for state in states:
        final_fault = history_has(state.history, HIST_FINAL_BASE, FAULTS)
        success = (
            state.done
            and final_fault == state.spec.fault
            and state.fixed_fault == state.spec.fault
            and state.memory_written == state.spec.fault
        )
        successes.append(1.0 if success else 0.0)
        success_by_type[EPISODE_NAMES[state.spec.episode_type]].append(1.0 if success else 0.0)
        tool_counts.append(sum(1 for token in state.history if token in (HIST_ACTION_BASE + ACTION_QUERY_LOG, HIST_ACTION_BASE + ACTION_RUN_TEST)))
    model.train()
    return {
        "episode_success": statistics.fmean(successes),
        "success_by_type": {name: statistics.fmean(values) if values else 0.0 for name, values in success_by_type.items()},
        "avg_tool_calls": statistics.fmean(tool_counts),
        "invalid_action_rate": invalid_actions / max(total_actions, 1),
        "extra_tool_rate": extra_tools / max(total_actions, 1),
    }


class LatentProbe(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fault = nn.Linear(d_model, FAULTS)
        self.phase = nn.Linear(d_model, 5)

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"fault": self.fault(latent), "phase": self.phase(latent)}


def train_latent_probe(
    model: AgentTransformer,
    train_states: list[AgentState],
    test_states: list[AgentState],
    config: AgentConfig,
    *,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    probe = LatentProbe(config.d_model).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.lr, weight_decay=0.0)
    rng = random.Random(config.seed + 50000)
    started = time.perf_counter()
    for _ in range(config.probe_steps):
        batch = random_batch(train_states, rng=rng, batch_size=config.batch_size, device=device)
        with torch.no_grad():
            output = model(batch, mode="latent_bottleneck", return_hidden=True)
            layout = output["layout"]
            hidden = output["hidden"]
            latent = hidden[:, layout.latent_start : layout.latent_end].mean(dim=1)
        logits = probe(latent.detach())
        loss = F.cross_entropy(logits["fault"], batch.fault) + F.cross_entropy(logits["phase"], batch.phase)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    totals = {"fault": 0.0, "phase": 0.0}
    count = 0
    with torch.no_grad():
        for start in range(0, len(test_states), config.batch_size):
            batch = make_batch(test_states[start : start + config.batch_size], device=device)
            output = model(batch, mode="latent_bottleneck", return_hidden=True)
            layout = output["layout"]
            hidden = output["hidden"]
            latent = hidden[:, layout.latent_start : layout.latent_end].mean(dim=1)
            logits = probe(latent)
            totals["fault"] += float((logits["fault"].argmax(dim=-1) == batch.fault).sum().detach().cpu())
            totals["phase"] += float((logits["phase"].argmax(dim=-1) == batch.phase).sum().detach().cpu())
            count += len(batch.action)
    return {
        "training_seconds": round(time.perf_counter() - started, 3),
        "accuracies": {key: value / count for key, value in totals.items()},
    }


def write_samples(episodes: list[EpisodeSpec], *, output_dir: Path, device: torch.device) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    states = [initial_state(spec) for spec in episodes[:8]]
    batch = make_batch(states, device=device)
    first_row = torch.cat([batch.image[idx] for idx in range(4)], dim=2)
    second_row = torch.cat([batch.image[idx] for idx in range(4, 8)], dim=2)
    write_png(output_dir / "agent_panel_grid.png", torch.cat([first_row, second_row], dim=1))


def sample_rollout_trace(
    model: AgentTransformer,
    episode: EpisodeSpec,
    config: AgentConfig,
    *,
    device: torch.device,
) -> list[str]:
    state = initial_state(episode)
    trace = []
    for _ in range(config.max_rollout_steps):
        batch = make_batch([state], device=device)
        with torch.no_grad():
            output = model(batch, mode="latent_bottleneck")
            action = int(output["logits"].argmax(dim=-1).detach().cpu()[0])
        trace.append(action_to_text(action))
        state = step_environment(state, action)
        if state.done:
            break
    return trace


def run_experiment(config: AgentConfig, output_path: Path) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_i_agent device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    train_episodes = generate_episodes(config.train_size, seed=config.seed + 1)
    val_episodes = generate_episodes(config.val_size, seed=config.seed + 2)
    test_episodes = generate_episodes(config.test_size, seed=config.seed + 3)
    train_states = unroll_expert(train_episodes)
    val_states = unroll_expert(val_episodes)
    test_states = unroll_expert(test_episodes)
    write_samples(test_episodes, output_dir=output_path.parent / "samples" / f"seed{config.seed}", device=device)

    direct = AgentTransformer(config)
    bottleneck = AgentTransformer(config)
    training = {
        "direct_agent": train_model(
            direct,
            train_states,
            val_states,
            config,
            steps=config.direct_steps,
            seed=config.seed + 10,
            device=device,
            mode="direct",
            label="direct_agent",
        ),
        "latent_bottleneck_agent": train_model(
            bottleneck,
            train_states,
            val_states,
            config,
            steps=config.bottleneck_steps,
            seed=config.seed + 20,
            device=device,
            mode="latent_bottleneck",
            label="latent_bottleneck_agent",
        ),
    }
    metrics = {
        "direct_agent": {
            **evaluate_steps(direct, test_states, config, device=device, mode="direct"),
            "rollout": rollout(direct, test_episodes, config, device=device, mode="direct"),
        },
        "latent_bottleneck_agent": {
            **evaluate_steps(bottleneck, test_states, config, device=device, mode="latent_bottleneck"),
            "rollout": rollout(bottleneck, test_episodes, config, device=device, mode="latent_bottleneck"),
        },
    }
    ablations = {
        "no_tool_history": rollout(bottleneck, test_episodes, config, device=device, mode="latent_bottleneck", zero_modalities=("history",)),
        "no_memory": rollout(bottleneck, test_episodes, config, device=device, mode="latent_bottleneck", zero_modalities=("memory",)),
        "no_image": rollout(bottleneck, test_episodes, config, device=device, mode="latent_bottleneck", zero_modalities=("image",)),
        "no_telemetry": rollout(bottleneck, test_episodes, config, device=device, mode="latent_bottleneck", zero_modalities=("telemetry",)),
        "no_latent_access": rollout(
            bottleneck,
            test_episodes,
            config,
            device=device,
            mode="latent_bottleneck",
            disable_latent_to_action=True,
        ),
    }
    latent_probe = train_latent_probe(bottleneck, train_states, test_states, config, device=device)

    output = {
        "experiment": "omni_transformer_stage_i_tool_using_agent",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "metrics": metrics,
        "ablations": ablations,
        "latent_probe": latent_probe,
        "training": training,
        "sample_rollouts": {
            EPISODE_NAMES[spec.episode_type]: sample_rollout_trace(bottleneck, spec, config, device=device)
            for spec in test_episodes[:8]
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {output_path}", flush=True)
    return output


def collect_numbers(value: object, prefix: tuple[str, ...] = ()) -> dict[str, float]:
    if isinstance(value, bool):
        return {}
    if isinstance(value, (int, float)):
        return {".".join(prefix): float(value)}
    if isinstance(value, dict):
        collected: dict[str, float] = {}
        for key, child in value.items():
            collected.update(collect_numbers(child, (*prefix, str(key))))
        return collected
    return {}


def aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    buckets: dict[str, list[float]] = {}
    for run in runs:
        for key, value in collect_numbers({"metrics": run["metrics"], "ablations": run["ablations"], "latent_probe": run["latent_probe"]}).items():
            if "training_seconds" not in key:
                buckets.setdefault(key, []).append(value)
    return {
        "experiment": "omni_transformer_stage_i_tool_using_agent_sweep",
        "run_count": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "stats": {key: stat(values) for key, values in sorted(buckets.items())},
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_i_agent/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_i_agent/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_i_agent/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--direct-steps", type=int, default=750)
    parser.add_argument("--bottleneck-steps", type=int, default=850)
    parser.add_argument("--probe-steps", type=int, default=200)
    args = parser.parse_args()

    base_config = AgentConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        direct_steps=args.direct_steps,
        bottleneck_steps=args.bottleneck_steps,
        probe_steps=args.probe_steps,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = AgentConfig(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
