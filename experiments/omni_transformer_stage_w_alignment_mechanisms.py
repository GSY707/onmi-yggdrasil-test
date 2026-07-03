from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import omni_transformer_stage_m_moe_multimodal_llm as stage_m
import omni_transformer_stage_q_tiny_moe_vlm_from_scratch as stage_q
import omni_transformer_stage_u_object_slots as stage_u
import omni_transformer_stage_v_expert_alignment as stage_v


MODES = (
    "ranking_only",
    "shared_grid_decoder",
    "shared_grid_slot_targets",
    "shared_slot_contrastive",
    "all_four_train_only_bus",
)
DIAGNOSTIC_STAGES = stage_v.DIAGNOSTIC_STAGES

MODE_SPECS = {
    "ranking_only": {"shared": False, "slot": False, "contrastive": False, "bus": False},
    "shared_grid_decoder": {"shared": True, "slot": False, "contrastive": False, "bus": False},
    "shared_grid_slot_targets": {"shared": True, "slot": True, "contrastive": False, "bus": False},
    "shared_slot_contrastive": {"shared": True, "slot": True, "contrastive": True, "bus": False},
    "all_four_train_only_bus": {"shared": True, "slot": True, "contrastive": True, "bus": True},
}


@dataclass(frozen=True)
class StageWConfig:
    train_size: int = 1024
    val_size: int = 256
    test_size: int = 384
    batch_size: int = 64
    seed: int = 20260701
    image_size: int = 64
    grid_size: int = 4
    prompt_len: int = 112
    answer_len: int = 32
    d_model: int = 96
    layers: int = 2
    heads: int = 4
    dropout: float = 0.0
    lr: float = 8e-4
    patch_grid: int = 8
    object_slots: int = 8
    spatial_tokens: int = 8
    count_tokens: int = 8
    prompt_tokens: int = 8
    fusion_tokens: int = 8
    output_tokens: int = 8
    candidate_count: int = 8
    train_steps: int = 360
    eval_every: int = 120
    router_loss_weight: float = 0.05
    shared_loss_weight: float = 0.30
    slot_loss_weight: float = 0.20
    contrastive_loss_weight: float = 0.08
    bus_loss_weight: float = 0.30
    bus_align_weight: float = 0.08
    probe_steps: int = 55
    probe_hidden: int = 192
    modes: tuple[str, ...] = MODES


def stage_v_config(config: StageWConfig) -> stage_v.StageVConfig:
    return stage_v.StageVConfig(
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        batch_size=config.batch_size,
        seed=config.seed,
        image_size=config.image_size,
        grid_size=config.grid_size,
        prompt_len=config.prompt_len,
        answer_len=config.answer_len,
        d_model=config.d_model,
        layers=config.layers,
        heads=config.heads,
        dropout=config.dropout,
        lr=config.lr,
        patch_grid=config.patch_grid,
        object_slots=config.object_slots,
        spatial_tokens=config.spatial_tokens,
        count_tokens=config.count_tokens,
        prompt_tokens=config.prompt_tokens,
        fusion_tokens=config.fusion_tokens,
        output_tokens=config.output_tokens,
        candidate_count=config.candidate_count,
        train_steps=config.train_steps,
        eval_every=config.eval_every,
        router_loss_weight=config.router_loss_weight,
        semantic_loss_weight=config.shared_loss_weight,
        probe_steps=config.probe_steps,
        probe_hidden=config.probe_hidden,
        modes=("ranking_only_object_latent",),
    )


class SlotObjectDecoder(nn.Module):
    def __init__(self, config: StageWConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
        )
        self.objectness = nn.Linear(config.d_model * 2, 1)
        self.color = nn.Linear(config.d_model * 2, len(stage_u.COLORS) + 1)
        self.shape = nn.Linear(config.d_model * 2, len(stage_u.SHAPES) + 1)
        self.cell = nn.Linear(config.d_model * 2, config.grid_size * config.grid_size + 1)

    def forward(self, object_slots: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.net(object_slots)
        return {
            "objectness": self.objectness(hidden).squeeze(-1),
            "color": self.color(hidden),
            "shape": self.shape(hidden),
            "cell": self.cell(hidden),
        }


def sorted_objects(meta_item: dict[str, object]) -> list[dict[str, int]]:
    objects = [dict(obj) for obj in meta_item["objects"]]
    objects.sort(key=lambda obj: (int(obj["row"]), int(obj["col"]), int(obj["shape"]), int(obj["color"])))
    return objects


def slot_targets(batch: stage_v.SemanticPixelSet, config: StageWConfig, *, device: torch.device) -> dict[str, torch.Tensor]:
    objectness = torch.zeros(len(batch.meta), config.object_slots, dtype=torch.float32, device=device)
    color = torch.zeros(len(batch.meta), config.object_slots, dtype=torch.long, device=device)
    shape = torch.zeros(len(batch.meta), config.object_slots, dtype=torch.long, device=device)
    cell = torch.zeros(len(batch.meta), config.object_slots, dtype=torch.long, device=device)
    for batch_index, meta_item in enumerate(batch.meta):
        for slot_index, obj in enumerate(sorted_objects(meta_item)[: config.object_slots]):
            objectness[batch_index, slot_index] = 1.0
            color[batch_index, slot_index] = int(obj["color"]) + 1
            shape[batch_index, slot_index] = int(obj["shape"]) + 1
            cell[batch_index, slot_index] = int(obj["row"]) * config.grid_size + int(obj["col"]) + 1
    return {"objectness": objectness, "color": color, "shape": shape, "cell": cell}


def slot_supervision_loss(decoder: SlotObjectDecoder, object_slots: torch.Tensor, batch: stage_v.SemanticPixelSet, config: StageWConfig) -> torch.Tensor:
    targets = slot_targets(batch, config, device=object_slots.device)
    output = decoder(object_slots)
    objectness_loss = F.binary_cross_entropy_with_logits(output["objectness"], targets["objectness"])
    positive = targets["objectness"].bool()
    if positive.any():
        color_loss = F.cross_entropy(output["color"][positive], targets["color"][positive])
        shape_loss = F.cross_entropy(output["shape"][positive], targets["shape"][positive])
        cell_loss = F.cross_entropy(output["cell"][positive], targets["cell"][positive])
    else:
        color_loss = output["color"].sum() * 0.0
        shape_loss = output["shape"].sum() * 0.0
        cell_loss = output["cell"].sum() * 0.0
    return objectness_loss + color_loss + shape_loss + cell_loss


@torch.no_grad()
def slot_metrics(decoder: SlotObjectDecoder, model: stage_u.StageUWideVLM, data: stage_v.SemanticPixelSet, config: StageWConfig, *, device: torch.device) -> dict[str, float]:
    model.eval()
    decoder.eval()
    rows = []
    for start in range(0, len(data.pixels.examples), config.batch_size):
        batch = data.subset(list(range(start, min(start + config.batch_size, len(data.pixels.examples))))).to(device)
        targets = slot_targets(batch, config, device=device)
        output = decoder(model.representations(batch.pixels)["object_slots"])
        objectness_pred = (torch.sigmoid(output["objectness"]) >= 0.5)
        positive = targets["objectness"].bool()
        row = {
            "objectness_accuracy": (objectness_pred == positive).float().mean().item(),
            "slot_exact": 0.0,
            "positive_color_accuracy": 0.0,
            "positive_shape_accuracy": 0.0,
            "positive_cell_accuracy": 0.0,
        }
        if positive.any():
            color_pred = output["color"].argmax(dim=-1)
            shape_pred = output["shape"].argmax(dim=-1)
            cell_pred = output["cell"].argmax(dim=-1)
            row["positive_color_accuracy"] = (color_pred[positive] == targets["color"][positive]).float().mean().item()
            row["positive_shape_accuracy"] = (shape_pred[positive] == targets["shape"][positive]).float().mean().item()
            row["positive_cell_accuracy"] = (cell_pred[positive] == targets["cell"][positive]).float().mean().item()
            exact = (
                objectness_pred
                & positive
                & (color_pred == targets["color"])
                & (shape_pred == targets["shape"])
                & (cell_pred == targets["cell"])
            ) | (~objectness_pred & ~positive)
            row["slot_exact"] = exact.float().mean().item()
        rows.append(row)
    return stage_v.merge_metric_rows(rows)


class StageProjector(nn.Module):
    def __init__(self, config: StageWConfig) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict({
            stage_name: nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model),
            )
            for stage_name in (*DIAGNOSTIC_STAGES, "common_bus")
        })
        self.temperature = 0.07

    def encode(self, stage_name: str, tokens: torch.Tensor) -> torch.Tensor:
        pooled = tokens.mean(dim=1)
        return F.normalize(self.projectors[stage_name](pooled), dim=-1)


def pairwise_contrastive(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    logits = z_a @ z_b.T / temperature
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def cross_expert_contrastive_loss(output: dict[str, torch.Tensor], projector: StageProjector, stages: Iterable[str] = DIAGNOSTIC_STAGES) -> torch.Tensor:
    stage_list = list(stages)
    embeddings = {stage_name: projector.encode(stage_name, output[stage_name]) for stage_name in stage_list}
    losses = []
    for left_index, left in enumerate(stage_list):
        for right in stage_list[left_index + 1 :]:
            losses.append(pairwise_contrastive(embeddings[left], embeddings[right], projector.temperature))
    return torch.stack(losses).mean()


class CommonSemanticBus(nn.Module):
    def __init__(self, config: StageWConfig) -> None:
        super().__init__()
        self.resampler = stage_q.QueryResampler(
            config.d_model,
            config.heads,
            config.grid_size * config.grid_size,
            config.dropout,
        )
        self.type_embedding = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return self.resampler(patch_tokens) + self.type_embedding


def common_bus_loss(
    common_bus: CommonSemanticBus,
    semantic_decoder: stage_v.GridSemanticDecoder,
    projector: StageProjector,
    output: dict[str, torch.Tensor],
    batch: stage_v.SemanticPixelSet,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    common_tokens = common_bus(output["patch_tokens"])
    semantic = stage_v.semantic_loss(semantic_decoder(common_tokens), batch)
    z_bus = projector.encode("common_bus", common_tokens)
    align_losses = [
        pairwise_contrastive(z_bus, projector.encode(stage_name, output[stage_name]), projector.temperature)
        for stage_name in DIAGNOSTIC_STAGES
    ]
    align = torch.stack(align_losses).mean()
    return semantic + align, semantic, align


@torch.no_grad()
def evaluate_common_bus(
    model: stage_u.StageUWideVLM,
    common_bus: CommonSemanticBus,
    semantic_decoder: stage_v.GridSemanticDecoder,
    data: stage_v.SemanticPixelSet,
    config: StageWConfig,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    common_bus.eval()
    semantic_decoder.eval()
    rows = []
    for start in range(0, len(data.pixels.examples), config.batch_size):
        batch = data.subset(list(range(start, min(start + config.batch_size, len(data.pixels.examples))))).to(device)
        reps = model.representations(batch.pixels)
        rows.append(stage_v.semantic_metrics(semantic_decoder(common_bus(reps["patch_tokens"])), batch))
    return stage_v.merge_metric_rows(rows)


def supervised_stage_loss(
    output: dict[str, torch.Tensor],
    decoder: stage_v.GridSemanticDecoder,
    batch: stage_v.SemanticPixelSet,
) -> torch.Tensor:
    return torch.stack([stage_v.semantic_loss(decoder(output[stage_name]), batch) for stage_name in DIAGNOSTIC_STAGES]).mean()


def train_mode(
    mode: str,
    train: stage_v.SemanticPixelSet,
    val: stage_v.SemanticPixelSet,
    config: StageWConfig,
    *,
    candidate_pool: stage_q.CandidatePool,
    seed: int,
    device: torch.device,
) -> tuple[
    stage_u.StageUWideVLM,
    stage_v.GridSemanticDecoder | None,
    SlotObjectDecoder | None,
    StageProjector | None,
    CommonSemanticBus | None,
    dict[str, object],
]:
    spec = MODE_SPECS[mode]
    rng = random.Random(seed)
    v_config = stage_v_config(config)
    u_config = stage_v.stage_u_config(v_config)
    model = stage_u.StageUWideVLM(u_config, variant="object_slot_spatial_wide_latent").to(device)
    semantic_decoder = stage_v.GridSemanticDecoder(v_config).to(device) if spec["shared"] or spec["bus"] else None
    slot_decoder = SlotObjectDecoder(config).to(device) if spec["slot"] else None
    projector = StageProjector(config).to(device) if spec["contrastive"] or spec["bus"] else None
    common_bus = CommonSemanticBus(config).to(device) if spec["bus"] else None
    modules: list[nn.Module] = [model]
    for module in (semantic_decoder, slot_decoder, projector, common_bus):
        if module is not None:
            modules.append(module)
    params = [parameter for module in modules for parameter in module.parameters()]
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    best_top1 = -1.0
    best_states: list[dict[str, torch.Tensor]] | None = None
    started = time.perf_counter()
    for step in range(1, config.train_steps + 1):
        batch = stage_v.random_semantic_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        rows, true_indices = stage_u.candidate_rows_for_examples(batch.pixels.examples, candidate_pool, seed=rng.randrange(1_000_000_000))
        answers = stage_q.answer_tokens_from_indices(candidate_pool, rows)
        labels = torch.tensor(true_indices, dtype=torch.long, device=device)
        output = model(batch.pixels, answers)
        rank_loss = stage_u.loss_for_output(output, labels, batch.pixels, model)
        shared_loss = torch.zeros((), device=device)
        slot_loss = torch.zeros((), device=device)
        contrastive_loss = torch.zeros((), device=device)
        bus_total_loss = torch.zeros((), device=device)
        bus_semantic_loss = torch.zeros((), device=device)
        bus_align_loss = torch.zeros((), device=device)
        loss = rank_loss
        if semantic_decoder is not None and spec["shared"]:
            shared_loss = supervised_stage_loss(output, semantic_decoder, batch)
            loss = loss + config.shared_loss_weight * shared_loss
        if slot_decoder is not None:
            slot_loss = slot_supervision_loss(slot_decoder, output["object_slots"], batch, config)
            loss = loss + config.slot_loss_weight * slot_loss
        if projector is not None and spec["contrastive"]:
            contrastive_loss = cross_expert_contrastive_loss(output, projector)
            loss = loss + config.contrastive_loss_weight * contrastive_loss
        if common_bus is not None and semantic_decoder is not None and projector is not None:
            bus_total_loss, bus_semantic_loss, bus_align_loss = common_bus_loss(common_bus, semantic_decoder, projector, output, batch)
            loss = loss + config.bus_loss_weight * bus_semantic_loss + config.bus_align_weight * bus_align_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.train_steps:
            metrics = stage_u.evaluate_ranking(
                model,
                val.pixels,
                u_config,
                device=device,
                candidate_pool=candidate_pool,
                seed=seed + step,
            )
            if float(metrics["rank_top1"]) > best_top1:
                best_top1 = float(metrics["rank_top1"])
                best_states = [
                    {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
                    for module in modules
                ]
            history.append(
                {
                    "step": step,
                    "loss": round(float(loss.detach().cpu()), 4),
                    "rank_loss": round(float(rank_loss.detach().cpu()), 4),
                    "shared_loss": round(float(shared_loss.detach().cpu()), 4),
                    "slot_loss": round(float(slot_loss.detach().cpu()), 4),
                    "contrastive_loss": round(float(contrastive_loss.detach().cpu()), 4),
                    "bus_semantic_loss": round(float(bus_semantic_loss.detach().cpu()), 4),
                    "bus_align_loss": round(float(bus_align_loss.detach().cpu()), 4),
                    "rank_top1": metrics["rank_top1"],
                }
            )
            print(
                f"{mode} step={step:4d} loss={float(loss.detach().cpu()):.4f} "
                f"rank={float(rank_loss.detach().cpu()):.4f} shared={float(shared_loss.detach().cpu()):.4f} "
                f"slot={float(slot_loss.detach().cpu()):.4f} contrast={float(contrastive_loss.detach().cpu()):.4f} "
                f"bus={float(bus_total_loss.detach().cpu()):.4f} top1={metrics['rank_top1']:.3f}",
                flush=True,
            )
    if best_states is not None:
        for module, state in zip(modules, best_states):
            module.load_state_dict(state)
    return (
        model,
        semantic_decoder,
        slot_decoder,
        projector,
        common_bus,
        {
            "history": history,
            "best_val_rank_top1": best_top1,
            "training_seconds": round(time.perf_counter() - started, 3),
            "trainable_parameter_count": sum(parameter.numel() for parameter in params if parameter.requires_grad),
        },
    )


def run_mode_diagnostics(
    mode: str,
    model: stage_u.StageUWideVLM,
    semantic_decoder: stage_v.GridSemanticDecoder | None,
    slot_decoder: SlotObjectDecoder | None,
    common_bus: CommonSemanticBus | None,
    train: stage_v.SemanticPixelSet,
    test: stage_v.SemanticPixelSet,
    config: StageWConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    v_config = stage_v_config(config)
    diagnostics = stage_v.run_diagnostics(
        model,
        semantic_decoder,
        train,
        test,
        v_config,
        seed=seed,
        device=device,
    )
    if slot_decoder is not None:
        diagnostics["slot_decoder"] = slot_metrics(slot_decoder, model, test, config, device=device)
    if common_bus is not None and semantic_decoder is not None:
        diagnostics["common_bus_train_only"] = evaluate_common_bus(model, common_bus, semantic_decoder, test, config, device=device)
    diagnostics["mechanisms"] = MODE_SPECS[mode]
    diagnostics["inference_uses_common_bus"] = False
    return diagnostics


def run_experiment(config: StageWConfig, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_w_alignment_mechanisms device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    v_config = stage_v_config(config)
    data, data_stats = stage_v.build_semantic_data(v_config)
    train = data["train"].to(device)
    val = data["val"].to(device)
    test = data["test"].to(device)
    stage_u.stage_r.write_samples(data["test"].pixels, data["test"].meta, output_path.parent / "samples")
    candidate_pool = stage_u.build_candidate_pool(stage_v.stage_u_config(v_config), device=device)

    metrics = {}
    training = {}
    diagnostics = {}
    for index, mode in enumerate(config.modes):
        model, semantic_decoder, slot_decoder, _projector, common_bus, train_stats = train_mode(
            mode,
            train,
            val,
            config,
            candidate_pool=candidate_pool,
            seed=config.seed + 1100 + index * 173,
            device=device,
        )
        training[mode] = train_stats
        metrics[mode] = stage_v.evaluate_model_bundle(
            model,
            test,
            v_config,
            candidate_pool=candidate_pool,
            device=device,
            seed=config.seed + 3100 + index * 19,
        )
        diagnostics[mode] = run_mode_diagnostics(
            mode,
            model,
            semantic_decoder,
            slot_decoder,
            common_bus,
            train,
            test,
            config,
            seed=config.seed + 6100 + index * 307,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = {
        "experiment": "omni_transformer_stage_w_alignment_mechanisms",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "architecture": {
            "base": "Stage U object_slot_spatial_wide_latent with Stage V semantic diagnostics",
            "mechanisms": {
                "shared_grid_decoder": "same semantic decoder reads all expert stages and reconstructs 4x4 occupancy/color/shape grid",
                "slot_targets": "object slots predict sorted objectness/color/shape/cell targets",
                "cross_expert_contrastive": "same-scene expert pooled embeddings are pulled together against in-batch negatives",
                "train_only_common_semantic_bus": "common bus is used only as an auxiliary training teacher/alignment target and is not read by the final scorer",
            },
            "inference_path": "candidate scorer still cross-attends direct wide latent; common semantic bus is not a runtime dependency",
            "mode_specs": MODE_SPECS,
        },
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "data": data_stats,
        "metrics": metrics,
        "training": training,
        "diagnostics": diagnostics,
        "total_seconds": round(time.perf_counter() - started, 3),
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
        for key, value in collect_numbers(
            {
                "metrics": run["metrics"],
                "training": run["training"],
                "diagnostics": run["diagnostics"],
                "total_seconds": run["total_seconds"],
            }
        ).items():
            buckets.setdefault(key, []).append(value)
    stats = {key: stage_m.stat(values) for key, values in sorted(buckets.items())}

    def mean(path: str) -> float | None:
        item = stats.get(path)
        return None if item is None else float(item["mean"])

    summary = {}
    for mode in MODES:
        if mode not in runs[0]["metrics"]:
            continue
        summary[mode] = {
            "rank_top1": mean(f"metrics.{mode}.full.rank_top1"),
            "no_image_top1": mean(f"metrics.{mode}.no_image_modality.rank_top1"),
            "no_patch_top1": mean(f"metrics.{mode}.no_patch_expert.rank_top1"),
            "no_object_top1": mean(f"metrics.{mode}.no_object_experts.rank_top1"),
            "train_seconds": mean(f"training.{mode}.training_seconds"),
            "params": mean(f"training.{mode}.trainable_parameter_count"),
            "diagonal_cell_info_accuracy": mean(f"diagnostics.{mode}.cross_expert_summary.diagonal_cell_info_accuracy"),
            "offdiag_cell_info_accuracy": mean(f"diagnostics.{mode}.cross_expert_summary.offdiag_cell_info_accuracy"),
            "language_gap_cell_info_accuracy": mean(f"diagnostics.{mode}.cross_expert_summary.language_gap_cell_info_accuracy"),
            "stage_cell_info_accuracy": {
                stage_name: mean(f"diagnostics.{mode}.diagonal_information_recovery.{stage_name}.cell_info_accuracy")
                for stage_name in DIAGNOSTIC_STAGES
            },
            "slot_decoder": {
                "slot_exact": mean(f"diagnostics.{mode}.slot_decoder.slot_exact"),
                "positive_cell_accuracy": mean(f"diagnostics.{mode}.slot_decoder.positive_cell_accuracy"),
                "positive_color_accuracy": mean(f"diagnostics.{mode}.slot_decoder.positive_color_accuracy"),
                "positive_shape_accuracy": mean(f"diagnostics.{mode}.slot_decoder.positive_shape_accuracy"),
            },
            "common_bus_cell_info_accuracy": mean(f"diagnostics.{mode}.common_bus_train_only.cell_info_accuracy"),
        }
    return {
        "experiment": "omni_transformer_stage_w_alignment_mechanisms_sweep",
        "run_count": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "stats": stats,
        "summary": summary,
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_w_alignment_mechanisms/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_w_alignment_mechanisms/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_w_alignment_mechanisms/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702")
    parser.add_argument("--modes", default="ranking_only,shared_grid_decoder,shared_grid_slot_targets,shared_slot_contrastive,all_four_train_only_bus")
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=112)
    parser.add_argument("--answer-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--patch-grid", type=int, default=8)
    parser.add_argument("--object-slots", type=int, default=8)
    parser.add_argument("--spatial-tokens", type=int, default=8)
    parser.add_argument("--count-tokens", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--fusion-tokens", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=360)
    parser.add_argument("--eval-every", type=int, default=120)
    parser.add_argument("--shared-loss-weight", type=float, default=0.30)
    parser.add_argument("--slot-loss-weight", type=float, default=0.20)
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.08)
    parser.add_argument("--bus-loss-weight", type=float, default=0.30)
    parser.add_argument("--bus-align-weight", type=float, default=0.08)
    parser.add_argument("--probe-steps", type=int, default=55)
    args = parser.parse_args()

    modes = parse_csv_strings(args.modes)
    unknown = set(modes) - set(MODES)
    if unknown:
        raise ValueError(f"unknown modes: {sorted(unknown)}")
    base_config = StageWConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        image_size=args.image_size,
        grid_size=args.grid_size,
        prompt_len=args.prompt_len,
        answer_len=args.answer_len,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        patch_grid=args.patch_grid,
        object_slots=args.object_slots,
        spatial_tokens=args.spatial_tokens,
        count_tokens=args.count_tokens,
        prompt_tokens=args.prompt_tokens,
        fusion_tokens=args.fusion_tokens,
        output_tokens=args.output_tokens,
        candidate_count=args.candidate_count,
        train_steps=args.train_steps,
        eval_every=args.eval_every,
        shared_loss_weight=args.shared_loss_weight,
        slot_loss_weight=args.slot_loss_weight,
        contrastive_loss_weight=args.contrastive_loss_weight,
        bus_loss_weight=args.bus_loss_weight,
        bus_align_weight=args.bus_align_weight,
        probe_steps=args.probe_steps,
        modes=modes,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = StageWConfig(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
