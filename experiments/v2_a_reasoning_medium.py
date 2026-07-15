from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from yggdrasil_v2.reasoning_medium.baseline import run_text_baselines
from yggdrasil_v2.reasoning_medium.data import DatasetSpec, build_dataset
from yggdrasil_v2.reasoning_medium.model import LatentConfig
from yggdrasil_v2.reasoning_medium.train import train_latent_reasoner

MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260712)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project-Yggdrasil V2-A reasoning-medium experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Generate deterministic register-machine splits")
    prepare.add_argument("--output-dir", type=Path, default=Path("artifacts/v2-a/data"))
    prepare.add_argument("--seed", type=int, default=20260712)
    prepare.add_argument("--train-size", type=int, default=512)
    prepare.add_argument("--validation-size", type=int, default=128)
    prepare.add_argument("--test-size", type=int, default=128)
    prepare.add_argument("--composition-size", type=int, default=128)
    prepare.add_argument("--length-size", type=int, default=128)

    baseline = subparsers.add_parser("baseline", help="Run deterministic direct/answer-only/text-CoT baselines")
    _add_model_arguments(baseline)
    baseline.add_argument("--data-dir", type=Path, default=Path("artifacts/v2-a/data"))
    baseline.add_argument("--output", type=Path, default=Path("artifacts/v2-a/a0/baseline_results.json"))
    baseline.add_argument("--split", default="test")
    baseline.add_argument("--max-examples", type=int, default=32)
    baseline.add_argument(
        "--modes",
        nargs="+",
        choices=("direct", "answer_only", "text_cot"),
        default=("direct", "answer_only", "text_cot"),
    )
    baseline.add_argument("--enable-thinking", action="store_true")

    latent = subparsers.add_parser("train-latent", help="Train a K-vector recurrent latent reasoner")
    _add_model_arguments(latent)
    latent.add_argument("--data-dir", type=Path, default=Path("artifacts/v2-a/data"))
    latent.add_argument("--output-dir", type=Path, default=Path("artifacts/v2-a/a1/k8-t8-seed20260712"))
    latent.add_argument("--latent-tokens", type=int, default=8)
    latent.add_argument("--recurrent-steps", type=int, default=8)
    latent.add_argument("--recurrent-blocks", type=int, default=2)
    latent.add_argument("--cross-attention-heads", type=int, default=8)
    latent.add_argument("--source-adapter-backend", choices=("none", "mlp"), default="none")
    latent.add_argument("--source-adapter-bottleneck", type=int, default=512)
    latent.add_argument("--steps", type=int, default=400)
    latent.add_argument("--batch-size", type=int, default=16)
    latent.add_argument("--encoder-batch-size", type=int, default=8)
    latent.add_argument("--learning-rate", type=float, default=2e-4)
    latent.add_argument("--max-length", type=int, default=256)
    latent.add_argument("--max-train-examples", type=int)
    latent.add_argument("--max-eval-examples", type=int)
    latent.add_argument("--no-resume", action="store_true")
    latent.add_argument("--teacher-hidden-weight", type=float, default=0.0)
    latent.add_argument("--state-supervision-weight", type=float, default=0.0)
    latent.add_argument("--encoder-demonstrations", type=int, default=0)
    latent.add_argument("--query-state-supervision-weight", type=float, default=0.0)
    latent.add_argument("--verifier-rl-weight", type=float, default=0.0)
    latent.add_argument("--source-mean-init", action="store_true")
    latent.add_argument("--transition-gate-init", type=float, default=-2.0)
    latent.add_argument("--transition-backend", choices=("copied_qwen", "mlp"), default="copied_qwen")
    latent.add_argument("--transition-bottleneck", type=int, default=512)
    latent.add_argument("--answer-pooling", choices=("mean", "flatten"), default="mean")
    latent.add_argument("--source-read-each-step", action="store_true")
    latent.add_argument("--source-read-gate-init", type=float, default=-2.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-data":
        result = build_dataset(
            args.output_dir,
            DatasetSpec(
                seed=args.seed,
                train_size=args.train_size,
                validation_size=args.validation_size,
                test_size=args.test_size,
                composition_size=args.composition_size,
                length_size=args.length_size,
            ),
        )
    elif args.command == "baseline":
        result = run_text_baselines(
            data_dir=args.data_dir,
            output_path=args.output,
            model_id=args.model_id,
            revision=args.revision,
            split=args.split,
            max_examples=args.max_examples,
            seed=args.seed,
            device=args.device,
            modes=tuple(args.modes),
            enable_thinking=args.enable_thinking,
        )
    else:
        result = train_latent_reasoner(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_id=args.model_id,
            revision=args.revision,
            seed=args.seed,
            device=args.device,
            latent_config=LatentConfig(
                latent_tokens=args.latent_tokens,
                recurrent_steps=args.recurrent_steps,
                recurrent_blocks=args.recurrent_blocks,
                cross_attention_heads=args.cross_attention_heads,
                source_adapter_backend=args.source_adapter_backend,
                source_adapter_bottleneck=args.source_adapter_bottleneck,
                source_mean_init=args.source_mean_init,
                transition_gate_init=args.transition_gate_init,
                transition_backend=args.transition_backend,
                transition_bottleneck=args.transition_bottleneck,
                answer_pooling=args.answer_pooling,
                source_read_each_step=args.source_read_each_step,
                source_read_gate_init=args.source_read_gate_init,
            ),
            steps=args.steps,
            batch_size=args.batch_size,
            encoder_batch_size=args.encoder_batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            max_train_examples=args.max_train_examples,
            max_eval_examples=args.max_eval_examples,
            resume=not args.no_resume,
            teacher_hidden_weight=args.teacher_hidden_weight,
            state_supervision_weight=args.state_supervision_weight,
            encoder_demonstrations=args.encoder_demonstrations,
            query_state_supervision_weight=args.query_state_supervision_weight,
            verifier_rl_weight=args.verifier_rl_weight,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
