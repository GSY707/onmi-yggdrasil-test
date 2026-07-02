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


CLASSES = 4
EPISODE_TYPES = 4
TEXT_LEN = 8
MEMORY_LEN = 4
HISTORY_LEN = 16
TELEMETRY_STEPS = 12
LATENT_TOKENS = 8

TASK_AUDIT = "audit"
TASK_UI_DOM = "ui_dom"
TASKS = (TASK_AUDIT, TASK_UI_DOM)

EP_AUDIT_CHART_TABLE = 0
EP_AUDIT_SCREEN_LOG = 1
EP_AUDIT_POLICY_CASE = 2
EP_AUDIT_MEMORY_AIDED = 3
AUDIT_EPISODE_NAMES = {
    EP_AUDIT_CHART_TABLE: "chart_table_mismatch",
    EP_AUDIT_SCREEN_LOG: "screenshot_log_contradiction",
    EP_AUDIT_POLICY_CASE: "policy_case_data",
    EP_AUDIT_MEMORY_AIDED: "memory_aided_incident",
}
AUDIT_FINAL_NAMES = ("EVIDENCE_0", "EVIDENCE_1", "EVIDENCE_2", "EVIDENCE_3")

EP_UI_BUTTON = 0
EP_UI_FORM = 1
EP_UI_TOGGLE = 2
EP_UI_DISABLED = 3
UI_EPISODE_NAMES = {
    EP_UI_BUTTON: "button_by_screenshot",
    EP_UI_FORM: "form_by_dom_label",
    EP_UI_TOGGLE: "toggle_by_state",
    EP_UI_DISABLED: "disabled_conflict",
}
UI_FINAL_NAMES = ("CLICK", "FORM", "TOGGLE", "DISABLED")

ACTION_TOOL_INSPECT = 0
ACTION_TOOL_QUERY = 1
ACTION_TOOL_READ = 2
ACTION_TOOL_VALIDATE = 3
ACTION_OPERATE_BASE = 4
ACTION_TYPE_BASE = ACTION_OPERATE_BASE + CLASSES
ACTION_MEMORY_BASE = ACTION_TYPE_BASE + CLASSES
ACTION_FINAL_BASE = ACTION_MEMORY_BASE + CLASSES
ACTION_FINAL_COUNT = CLASSES * EPISODE_TYPES
ACTION_VOCAB = ACTION_FINAL_BASE + ACTION_FINAL_COUNT

HIST_PAD = 0
HIST_ACTION_BASE = 1
HIST_IMAGE_BASE = HIST_ACTION_BASE + ACTION_VOCAB
HIST_QUERY_BASE = HIST_IMAGE_BASE + CLASSES
HIST_READ_BASE = HIST_QUERY_BASE + CLASSES
HIST_VALIDATE_BASE = HIST_READ_BASE + CLASSES
HIST_OPERATE_BASE = HIST_VALIDATE_BASE + CLASSES
HIST_TYPE_BASE = HIST_OPERATE_BASE + CLASSES
HIST_MEMORY_BASE = HIST_TYPE_BASE + CLASSES
HIST_FINAL_BASE = HIST_MEMORY_BASE + CLASSES
HISTORY_VOCAB = HIST_FINAL_BASE + ACTION_FINAL_COUNT

TEXT_VOCAB = 80
MEMORY_VOCAB = 16
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
    direct_steps: int = 650
    bottleneck_steps: int = 750
    probe_steps: int = 180
    eval_every: int = 375
    max_rollout_steps: int = 7


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: int
    task: str
    episode_type: int
    target: int
    visual_cue: int
    structured_cue: int
    policy_or_state_cue: int
    memory_cue: int
    final_kind: int


@dataclass(frozen=True)
class AgentState:
    spec: EpisodeSpec
    history: tuple[int, ...]
    operated_target: int
    typed_target: int
    validated_target: int
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
    target: torch.Tensor
    phase: torch.Tensor
    final_kind: torch.Tensor
    episode_type: torch.Tensor


@dataclass(frozen=True)
class AgentLayout:
    input_end: int
    latent_start: int
    latent_end: int
    action_pos: int
    seq_len: int


def episode_names(task: str) -> dict[int, str]:
    return AUDIT_EPISODE_NAMES if task == TASK_AUDIT else UI_EPISODE_NAMES


def final_names(task: str) -> tuple[str, ...]:
    return AUDIT_FINAL_NAMES if task == TASK_AUDIT else UI_FINAL_NAMES


def final_action(target: int, kind: int) -> int:
    return ACTION_FINAL_BASE + target * EPISODE_TYPES + kind


def decode_final(action: int) -> tuple[int, int]:
    value = int(action) - ACTION_FINAL_BASE
    return value // EPISODE_TYPES, value % EPISODE_TYPES


def action_to_text(task: str, action: int) -> str:
    action = int(action)
    if task == TASK_AUDIT:
        tool_names = {
            ACTION_TOOL_INSPECT: "TOOL_INSPECT_IMAGE",
            ACTION_TOOL_QUERY: "TOOL_QUERY_TABLE",
            ACTION_TOOL_READ: "TOOL_READ_POLICY_OR_LOG",
            ACTION_TOOL_VALIDATE: "TOOL_CROSS_CHECK",
        }
        operate = "RECORD_FINDING"
    else:
        tool_names = {
            ACTION_TOOL_INSPECT: "TOOL_INSPECT_SCREEN",
            ACTION_TOOL_QUERY: "TOOL_QUERY_DOM",
            ACTION_TOOL_READ: "TOOL_READ_DOM_STATE",
            ACTION_TOOL_VALIDATE: "TOOL_VALIDATE_UI",
        }
        operate = "CLICK_ELEMENT"
    if action in tool_names:
        return tool_names[action]
    if ACTION_OPERATE_BASE <= action < ACTION_OPERATE_BASE + CLASSES:
        return f"{operate}_{action - ACTION_OPERATE_BASE}"
    if ACTION_TYPE_BASE <= action < ACTION_TYPE_BASE + CLASSES:
        return f"TYPE_FIELD_{action - ACTION_TYPE_BASE}"
    if ACTION_MEMORY_BASE <= action < ACTION_MEMORY_BASE + CLASSES:
        return f"MEMORY_WRITE_{action - ACTION_MEMORY_BASE}"
    if ACTION_FINAL_BASE <= action < ACTION_FINAL_BASE + ACTION_FINAL_COUNT:
        target, kind = decode_final(action)
        return f"FINAL_{target}_{final_names(task)[kind]}"
    return f"INVALID_{action}"


def history_tokens(history: tuple[int, ...]) -> tuple[int, ...]:
    trimmed = history[-HISTORY_LEN:]
    return trimmed + (HIST_PAD,) * (HISTORY_LEN - len(trimmed))


def history_has(history: tuple[int, ...], start: int, count: int) -> int:
    for token in history:
        if start <= token < start + count:
            return token - start
    return -1


def has_history_token(history: tuple[int, ...], token: int) -> bool:
    return token in history


def make_audit_episode(index: int, *, rng: random.Random) -> EpisodeSpec:
    episode_type = index % EPISODE_TYPES
    target = rng.randrange(CLASSES)
    visual_cue = rng.randrange(CLASSES)
    structured_cue = rng.randrange(CLASSES)
    policy_or_state_cue = rng.randrange(CLASSES)
    memory_cue = (target + 2) % CLASSES
    final_kind = episode_type
    if episode_type == EP_AUDIT_CHART_TABLE:
        target = visual_cue
        final_kind = structured_cue
    elif episode_type == EP_AUDIT_SCREEN_LOG:
        target = structured_cue
        policy_or_state_cue = structured_cue
        final_kind = visual_cue
    elif episode_type == EP_AUDIT_POLICY_CASE:
        structured_cue = rng.randrange(CLASSES)
        policy_or_state_cue = (target - structured_cue) % CLASSES
        visual_cue = (target + 1 + rng.randrange(CLASSES - 1)) % CLASSES
        final_kind = policy_or_state_cue
    else:
        visual_cue = (target + 1 + rng.randrange(CLASSES - 1)) % CLASSES
        structured_cue = (target + 1) % CLASSES
        policy_or_state_cue = (target + 3) % CLASSES
        memory_cue = target
        final_kind = structured_cue
    return EpisodeSpec(
        episode_id=index,
        task=TASK_AUDIT,
        episode_type=episode_type,
        target=target,
        visual_cue=visual_cue,
        structured_cue=structured_cue,
        policy_or_state_cue=policy_or_state_cue,
        memory_cue=memory_cue,
        final_kind=final_kind,
    )


def make_ui_episode(index: int, *, rng: random.Random) -> EpisodeSpec:
    episode_type = index % EPISODE_TYPES
    target = rng.randrange(CLASSES)
    visual_cue = target
    structured_cue = target
    policy_or_state_cue = target
    memory_cue = rng.randrange(CLASSES)
    if episode_type == EP_UI_FORM:
        visual_cue = (target + 1) % CLASSES
    elif episode_type == EP_UI_DISABLED:
        visual_cue = (target + 1 + rng.randrange(CLASSES - 1)) % CLASSES
    return EpisodeSpec(
        episode_id=index,
        task=TASK_UI_DOM,
        episode_type=episode_type,
        target=target,
        visual_cue=visual_cue,
        structured_cue=structured_cue,
        policy_or_state_cue=policy_or_state_cue,
        memory_cue=memory_cue,
        final_kind=episode_type,
    )


def generate_episodes(task: str, size: int, *, seed: int) -> list[EpisodeSpec]:
    rng = random.Random(seed)
    maker = make_audit_episode if task == TASK_AUDIT else make_ui_episode
    episodes = [maker(index, rng=rng) for index in range(size)]
    rng.shuffle(episodes)
    return episodes


def initial_state(spec: EpisodeSpec) -> AgentState:
    return AgentState(
        spec=spec,
        history=(),
        operated_target=-1,
        typed_target=-1,
        validated_target=-1,
        memory_written=-1,
        done=False,
        step=0,
    )


def audit_required_tools(state: AgentState) -> tuple[int, ...]:
    if state.spec.episode_type == EP_AUDIT_CHART_TABLE:
        return (ACTION_TOOL_INSPECT, ACTION_TOOL_QUERY)
    if state.spec.episode_type == EP_AUDIT_SCREEN_LOG:
        return (ACTION_TOOL_INSPECT, ACTION_TOOL_READ)
    if state.spec.episode_type == EP_AUDIT_POLICY_CASE:
        return (ACTION_TOOL_READ, ACTION_TOOL_QUERY)
    return (ACTION_TOOL_READ,)


def ui_required_tools(state: AgentState) -> tuple[int, ...]:
    if state.spec.episode_type == EP_UI_DISABLED:
        return (ACTION_TOOL_INSPECT, ACTION_TOOL_QUERY, ACTION_TOOL_READ)
    if state.spec.episode_type in (EP_UI_BUTTON, EP_UI_TOGGLE):
        return (ACTION_TOOL_INSPECT, ACTION_TOOL_QUERY)
    return (ACTION_TOOL_QUERY,)


def required_tools(state: AgentState) -> tuple[int, ...]:
    return audit_required_tools(state) if state.spec.task == TASK_AUDIT else ui_required_tools(state)


def tool_action_seen(state: AgentState, action: int) -> bool:
    return has_history_token(state.history, HIST_ACTION_BASE + action)


def phase_for(state: AgentState) -> int:
    if state.done:
        return 5
    for tool in required_tools(state):
        if not tool_action_seen(state, tool):
            return 0
    if state.spec.task == TASK_UI_DOM and state.spec.episode_type == EP_UI_FORM and state.typed_target < 0:
        return 1
    if state.operated_target < 0:
        return 2
    if state.spec.task == TASK_UI_DOM and state.validated_target < 0:
        return 3
    if state.memory_written < 0:
        return 4
    return 5


def expert_action(state: AgentState) -> int:
    target = state.spec.target
    for tool in required_tools(state):
        if not tool_action_seen(state, tool):
            return tool
    if state.spec.task == TASK_UI_DOM and state.spec.episode_type == EP_UI_FORM and state.typed_target < 0:
        return ACTION_TYPE_BASE + target
    if state.operated_target < 0:
        return ACTION_OPERATE_BASE + target
    if state.spec.task == TASK_UI_DOM and state.validated_target < 0:
        return ACTION_TOOL_VALIDATE
    if state.memory_written < 0:
        return ACTION_MEMORY_BASE + target
    return final_action(target, state.spec.final_kind)


def step_environment(state: AgentState, action: int) -> AgentState:
    history = [token for token in state.history if token != HIST_PAD]
    operated_target = state.operated_target
    typed_target = state.typed_target
    validated_target = state.validated_target
    memory_written = state.memory_written
    done = state.done

    history.append(HIST_ACTION_BASE + int(action))
    if action == ACTION_TOOL_INSPECT:
        history.append(HIST_IMAGE_BASE + state.spec.visual_cue)
    elif action == ACTION_TOOL_QUERY:
        history.append(HIST_QUERY_BASE + state.spec.structured_cue)
    elif action == ACTION_TOOL_READ:
        history.append(HIST_READ_BASE + state.spec.policy_or_state_cue)
    elif action == ACTION_TOOL_VALIDATE:
        if operated_target == state.spec.target and (
            state.spec.task == TASK_AUDIT
            or state.spec.episode_type != EP_UI_FORM
            or typed_target == state.spec.target
        ):
            validated_target = state.spec.target
        history.append(HIST_VALIDATE_BASE + max(validated_target, 0))
    elif ACTION_OPERATE_BASE <= action < ACTION_OPERATE_BASE + CLASSES:
        operated = action - ACTION_OPERATE_BASE
        if operated == state.spec.target:
            operated_target = operated
        history.append(HIST_OPERATE_BASE + operated)
    elif ACTION_TYPE_BASE <= action < ACTION_TYPE_BASE + CLASSES:
        typed = action - ACTION_TYPE_BASE
        if typed == state.spec.target:
            typed_target = typed
        history.append(HIST_TYPE_BASE + typed)
    elif ACTION_MEMORY_BASE <= action < ACTION_MEMORY_BASE + CLASSES:
        written = action - ACTION_MEMORY_BASE
        if written == state.spec.target:
            memory_written = written
        history.append(HIST_MEMORY_BASE + written)
    elif ACTION_FINAL_BASE <= action < ACTION_FINAL_BASE + ACTION_FINAL_COUNT:
        done = True
        history.append(HIST_FINAL_BASE + (action - ACTION_FINAL_BASE))

    return AgentState(
        spec=state.spec,
        history=tuple(history[-HISTORY_LEN:]),
        operated_target=operated_target,
        typed_target=typed_target,
        validated_target=validated_target,
        memory_written=memory_written,
        done=done,
        step=state.step + 1,
    )


def unroll_expert(episodes: list[EpisodeSpec], *, max_steps: int) -> list[AgentState]:
    states: list[AgentState] = []
    for spec in episodes:
        state = initial_state(spec)
        for _ in range(max_steps):
            states.append(state)
            action = expert_action(state)
            state = step_environment(state, action)
            if state.done:
                break
    return states


def render_ui_images(visual_ids: torch.Tensor, episode_types: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    batch = int(visual_ids.shape[0])
    image = torch.full((batch, 3, IMAGE_SIZE, IMAGE_SIZE), 0.08, device=device)
    image[:, :, 3:-3, 3:-3] = 0.14
    positions = ((7, 7), (7, 27), (27, 7), (27, 27))
    base_colors = torch.tensor(
        [
            [0.20, 0.62, 0.95],
            [0.95, 0.38, 0.30],
            [0.26, 0.78, 0.38],
            [0.95, 0.76, 0.18],
        ],
        device=device,
    )
    for idx, (y, x) in enumerate(positions):
        image[:, :, y : y + 14, x : x + 14] = base_colors[idx].view(1, 3, 1, 1) * 0.62
    for row in range(batch):
        target = int(visual_ids[row].detach().cpu())
        y, x = positions[target]
        image[row, :, y : y + 14, x : x + 14] = base_colors[target].view(3, 1, 1)
        if int(episode_types[row].detach().cpu()) == EP_UI_DISABLED:
            image[row, :, y + 5 : y + 9, x : x + 14] = 0.03
            image[row, :, y : y + 14, x + 5 : x + 9] = 0.03
    return image


def render_text_tokens(states: list[AgentState], *, device: torch.device) -> torch.Tensor:
    tokens = torch.zeros((len(states), TEXT_LEN), dtype=torch.long, device=device)
    for idx, state in enumerate(states):
        task_offset = 0 if state.spec.task == TASK_AUDIT else 1
        if state.spec.task == TASK_AUDIT and state.spec.episode_type == EP_AUDIT_POLICY_CASE:
            policy_or_goal = state.spec.policy_or_state_cue
        else:
            policy_or_goal = 4 + state.spec.episode_type
        tokens[idx, 0] = 1
        tokens[idx, 1] = 3 + task_offset
        tokens[idx, 2] = 8 + state.spec.episode_type
        tokens[idx, 3] = 16 + policy_or_goal
        tokens[idx, 4] = 24 + state.spec.episode_type
        tokens[idx, 5] = 32 + task_offset
        tokens[idx, 6] = 40 + state.spec.episode_type
        tokens[idx, 7] = 2
    return tokens


def render_memory_tokens(states: list[AgentState], *, device: torch.device) -> torch.Tensor:
    tokens = torch.zeros((len(states), MEMORY_LEN), dtype=torch.long, device=device)
    for idx, state in enumerate(states):
        tokens[idx, 0] = 1 + state.spec.memory_cue
        tokens[idx, 1] = 6 + state.spec.episode_type % 2
        tokens[idx, 2] = 8 + (0 if state.spec.task == TASK_AUDIT else 1)
        tokens[idx, 3] = 10 + state.spec.final_kind % 4
    return tokens


def make_batch(states: list[AgentState], *, device: torch.device) -> AgentBatch:
    visual = torch.tensor([state.spec.visual_cue for state in states], dtype=torch.long, device=device)
    episode_type = torch.tensor([state.spec.episode_type for state in states], dtype=torch.long, device=device)
    task = states[0].spec.task if states else TASK_AUDIT
    if task == TASK_UI_DOM:
        image = render_ui_images(visual, episode_type, device=device)
    else:
        variant = torch.tensor([(state.spec.episode_id + state.step) % 7 for state in states], dtype=torch.long, device=device)
        image = render_augmented_panel_images(visual, device=device, variant_ids=variant)

    structured = torch.tensor([state.spec.structured_cue for state in states], dtype=torch.long, device=device)
    return AgentBatch(
        image=image,
        text=render_text_tokens(states, device=device),
        telemetry=render_telemetry(structured, device=device),
        memory=render_memory_tokens(states, device=device),
        history=torch.tensor([history_tokens(state.history) for state in states], dtype=torch.long, device=device),
        action=torch.tensor([expert_action(state) for state in states], dtype=torch.long, device=device),
        target=torch.tensor([state.spec.target for state in states], dtype=torch.long, device=device),
        phase=torch.tensor([phase_for(state) for state in states], dtype=torch.long, device=device),
        final_kind=torch.tensor([state.spec.final_kind for state in states], dtype=torch.long, device=device),
        episode_type=episode_type,
    )


def random_batch(states: list[AgentState], *, rng: random.Random, batch_size: int, device: torch.device) -> AgentBatch:
    return make_batch(rng.choices(states, k=batch_size), device=device)


def mask_history_range(history: torch.Tensor, start: int, count: int) -> torch.Tensor:
    masked = history.clone()
    keep = (masked < start) | (masked >= start + count)
    return torch.where(keep, masked, torch.zeros_like(masked))


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
        self.position_embedding = nn.Parameter(torch.randn(160, config.d_model) * 0.02)
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
        return AgentLayout(
            input_end=input_end,
            latent_start=latent_start,
            latent_end=latent_end,
            action_pos=action_pos,
            seq_len=action_pos + 1,
        )

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
        history_input = batch.history
        if "image" in zero_set:
            image_tokens = torch.zeros_like(image_tokens)
            history_input = mask_history_range(history_input, HIST_IMAGE_BASE, CLASSES)
        if "text" in zero_set:
            text_tokens = torch.zeros_like(text_tokens)
        if "telemetry" in zero_set or "dom" in zero_set or "structured" in zero_set:
            telemetry_tokens = torch.zeros_like(telemetry_tokens)
            history_input = mask_history_range(history_input, HIST_QUERY_BASE, CLASSES)
            history_input = mask_history_range(history_input, HIST_READ_BASE, CLASSES)
            history_input = mask_history_range(history_input, HIST_VALIDATE_BASE, CLASSES)
        if "memory" in zero_set:
            memory_tokens = torch.zeros_like(memory_tokens)
            history_input = mask_history_range(history_input, HIST_MEMORY_BASE, CLASSES)
        if "history" in zero_set:
            history_input = torch.zeros_like(history_input)
        history_tokens_emb = self.history_embedding(history_input)

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
            history_tokens_emb,
            self.latent_embedding.expand(batch_size, LATENT_TOKENS, -1),
            special[SPECIAL_ACTION].expand(batch_size, 1, -1),
        ]
        x = torch.cat(chunks, dim=1)
        x = x + self.position_embedding[: x.shape[1]].unsqueeze(0)
        return x, self.layout()

    def attention_mask(
        self,
        layout: AgentLayout,
        *,
        mode: str,
        disable_latent_to_action: bool,
        device: torch.device,
    ) -> torch.Tensor:
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
        mask = self.attention_mask(
            layout,
            mode=mode,
            disable_latent_to_action=disable_latent_to_action,
            device=x.device,
        )
        for block in self.blocks:
            x = block(x, mask)
        hidden = self.norm(x)
        output: dict[str, torch.Tensor | AgentLayout] = {
            "logits": self.action_head(hidden[:, layout.action_pos]),
            "layout": layout,
        }
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
            print(
                f"{label} step={step:4d} loss={float(loss.detach().cpu()):.4f} "
                f"step_exact={metrics['step_action_exact']:.3f}",
                flush=True,
            )
    return {
        "history": history,
        "training_seconds": round(time.perf_counter() - started, 3),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def final_from_history(history: tuple[int, ...]) -> tuple[int, int]:
    final = history_has(history, HIST_FINAL_BASE, ACTION_FINAL_COUNT)
    if final < 0:
        return -1, -1
    return final // EPISODE_TYPES, final % EPISODE_TYPES


def state_success(state: AgentState) -> bool:
    final_target, final_kind = final_from_history(state.history)
    if state.spec.task == TASK_AUDIT:
        return (
            state.done
            and final_target == state.spec.target
            and final_kind == state.spec.final_kind
            and state.operated_target == state.spec.target
            and state.memory_written == state.spec.target
        )
    typed_ok = state.spec.episode_type != EP_UI_FORM or state.typed_target == state.spec.target
    return (
        state.done
        and final_target == state.spec.target
        and final_kind == state.spec.final_kind
        and state.operated_target == state.spec.target
        and state.validated_target == state.spec.target
        and state.memory_written == state.spec.target
        and typed_ok
    )


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
            states[idx] = step_environment(states[idx], int(action))
            finished[idx] = states[idx].done

    names = episode_names(episodes[0].task if episodes else TASK_AUDIT)
    success_by_type: dict[str, list[float]] = {name: [] for name in names.values()}
    successes: list[float] = []
    target_exact: list[float] = []
    kind_exact: list[float] = []
    for state in states:
        final_target, final_kind = final_from_history(state.history)
        success = state_success(state)
        successes.append(1.0 if success else 0.0)
        success_by_type[names[state.spec.episode_type]].append(1.0 if success else 0.0)
        target_exact.append(1.0 if final_target == state.spec.target else 0.0)
        kind_exact.append(1.0 if final_kind == state.spec.final_kind else 0.0)
    model.train()
    return {
        "episode_success": statistics.fmean(successes),
        "success_by_type": {name: statistics.fmean(values) if values else 0.0 for name, values in success_by_type.items()},
        "final_target_exact": statistics.fmean(target_exact),
        "final_kind_exact": statistics.fmean(kind_exact),
        "invalid_action_rate": invalid_actions / max(total_actions, 1),
        "avg_rollout_steps": total_actions / max(len(episodes), 1),
    }


class LatentProbe(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.target = nn.Linear(d_model, CLASSES)
        self.phase = nn.Linear(d_model, 6)
        self.final_kind = nn.Linear(d_model, EPISODE_TYPES)

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "target": self.target(latent),
            "phase": self.phase(latent),
            "final_kind": self.final_kind(latent),
        }


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
        loss = (
            F.cross_entropy(logits["target"], batch.target)
            + F.cross_entropy(logits["phase"], batch.phase)
            + F.cross_entropy(logits["final_kind"], batch.final_kind)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    totals = {"target": 0.0, "phase": 0.0, "final_kind": 0.0}
    count = 0
    with torch.no_grad():
        for start in range(0, len(test_states), config.batch_size):
            batch = make_batch(test_states[start : start + config.batch_size], device=device)
            output = model(batch, mode="latent_bottleneck", return_hidden=True)
            layout = output["layout"]
            hidden = output["hidden"]
            latent = hidden[:, layout.latent_start : layout.latent_end].mean(dim=1)
            logits = probe(latent)
            totals["target"] += float((logits["target"].argmax(dim=-1) == batch.target).sum().detach().cpu())
            totals["phase"] += float((logits["phase"].argmax(dim=-1) == batch.phase).sum().detach().cpu())
            totals["final_kind"] += float((logits["final_kind"].argmax(dim=-1) == batch.final_kind).sum().detach().cpu())
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
    write_png(output_dir / "agent_observation_grid.png", torch.cat([first_row, second_row], dim=1))


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
        trace.append(action_to_text(episode.task, action))
        state = step_environment(state, action)
        if state.done:
            break
    return trace


def run_task_experiment(task: str, config: AgentConfig, output_path: Path) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_jk task={task} device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    train_episodes = generate_episodes(task, config.train_size, seed=config.seed + 1)
    val_episodes = generate_episodes(task, config.val_size, seed=config.seed + 2)
    test_episodes = generate_episodes(task, config.test_size, seed=config.seed + 3)
    train_states = unroll_expert(train_episodes, max_steps=config.max_rollout_steps)
    val_states = unroll_expert(val_episodes, max_steps=config.max_rollout_steps)
    test_states = unroll_expert(test_episodes, max_steps=config.max_rollout_steps)
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
            label=f"{task}_direct",
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
            label=f"{task}_latent",
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
    if task == TASK_AUDIT:
        ablations = {
            "no_tool_history": rollout(bottleneck, test_episodes, config, device=device, mode="latent_bottleneck", zero_modalities=("history",)),
            "no_memory": rollout(bottleneck, test_episodes, config, device=device, mode="latent_bottleneck", zero_modalities=("memory",)),
            "no_image": rollout(bottleneck, test_episodes, config, device=device, mode="latent_bottleneck", zero_modalities=("image",)),
            "no_structured_table_log": rollout(bottleneck, test_episodes, config, device=device, mode="latent_bottleneck", zero_modalities=("structured",)),
            "no_policy_text": rollout(bottleneck, test_episodes, config, device=device, mode="latent_bottleneck", zero_modalities=("text",)),
            "no_latent_access": rollout(
                bottleneck,
                test_episodes,
                config,
                device=device,
                mode="latent_bottleneck",
                disable_latent_to_action=True,
            ),
        }
    else:
        ablations = {
            "no_tool_history": rollout(bottleneck, test_episodes, config, device=device, mode="latent_bottleneck", zero_modalities=("history",)),
            "no_dom_structured": rollout(bottleneck, test_episodes, config, device=device, mode="latent_bottleneck", zero_modalities=("dom",)),
            "no_image": rollout(bottleneck, test_episodes, config, device=device, mode="latent_bottleneck", zero_modalities=("image",)),
            "no_goal_text": rollout(bottleneck, test_episodes, config, device=device, mode="latent_bottleneck", zero_modalities=("text",)),
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

    names = episode_names(task)
    examples_by_type: dict[str, list[str]] = {}
    seen_types: set[int] = set()
    for spec in test_episodes:
        if spec.episode_type in seen_types:
            continue
        examples_by_type[names[spec.episode_type]] = sample_rollout_trace(bottleneck, spec, config, device=device)
        seen_types.add(spec.episode_type)
        if len(seen_types) == EPISODE_TYPES:
            break

    output = {
        "experiment": "omni_transformer_stage_jk_audit_ui",
        "task": task,
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
        "sample_rollouts": examples_by_type,
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


def aggregate_runs(task: str, runs: list[dict[str, object]]) -> dict[str, object]:
    buckets: dict[str, list[float]] = {}
    for run in runs:
        numbers = collect_numbers({"metrics": run["metrics"], "ablations": run["ablations"], "latent_probe": run["latent_probe"]})
        for key, value in numbers.items():
            if "training_seconds" not in key:
                buckets.setdefault(key, []).append(value)
    return {
        "task": task,
        "run_count": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "stats": {key: stat(values) for key, values in sorted(buckets.items())},
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_tasks(value: str) -> tuple[str, ...]:
    if value == "both":
        return TASKS
    if value not in TASKS:
        raise ValueError(f"unknown task: {value}")
    return (value,)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("audit", "ui_dom", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_jk_audit_ui/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_jk_audit_ui/sweep_results.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_jk_audit_ui/result.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--direct-steps", type=int, default=650)
    parser.add_argument("--bottleneck-steps", type=int, default=750)
    parser.add_argument("--probe-steps", type=int, default=180)
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
    tasks = parse_tasks(args.task)
    if args.sweep:
        aggregate = {"experiment": "omni_transformer_stage_jk_audit_ui_sweep", "tasks": {}}
        for task in tasks:
            runs = []
            for seed in parse_csv_ints(args.seeds):
                config = AgentConfig(**{**asdict(base_config), "seed": seed})
                output_path = args.output_dir / f"{task}_seed{seed}" / "result.json"
                runs.append(run_task_experiment(task, config, output_path))
            aggregate["tasks"][task] = aggregate_runs(task, runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        for task in tasks:
            output_path = args.output if len(tasks) == 1 else args.output.with_name(f"{task}_{args.output.name}")
            run_task_experiment(task, base_config, output_path)


if __name__ == "__main__":
    main()
