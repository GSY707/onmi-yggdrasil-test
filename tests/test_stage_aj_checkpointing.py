import importlib.util
from dataclasses import asdict
from pathlib import Path
import sys


def load_stage_aj_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "omni_transformer_stage_aj_transformer_image_io_fidelity.py"
    spec = importlib.util.spec_from_file_location("stage_aj_checkpointing", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_checkpoint_config_compatibility_allows_runtime_controls():
    stage_aj = load_stage_aj_module()
    config = stage_aj.StageAJConfig(
        train_size=16,
        val_size=8,
        test_size=8,
        copy_steps=8,
        max_train_seconds=600,
        checkpoint_every=2,
        checkpoint_sample_every=2,
        variants=("memory_tree_supervised_copy",),
    )
    saved = asdict(config)
    saved["copy_steps"] = 4
    saved["max_train_seconds"] = 30
    saved["checkpoint_every"] = 1
    saved["variants"] = ("memory_tree_copy",)

    assert stage_aj.checkpoint_config_mismatches(saved, config) == {}

    saved["d_model"] = config.d_model + 1

    assert "d_model" in stage_aj.checkpoint_config_mismatches(saved, config)


def test_resolve_device_cpu_is_explicit():
    stage_aj = load_stage_aj_module()

    assert str(stage_aj.resolve_device("cpu")) == "cpu"
