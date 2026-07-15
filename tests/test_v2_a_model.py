from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from yggdrasil_v2.reasoning_medium.model import LatentConfig, LatentReasoner, assert_no_answer_bypass


def test_latent_reasoner_shapes_and_interventions() -> None:
    backbone = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )
    reasoner = LatentReasoner(
        backbone,
        LatentConfig(
            latent_tokens=4,
            recurrent_steps=2,
            recurrent_blocks=2,
            cross_attention_heads=4,
        ),
    )
    source = torch.randn(3, 12, 32)
    mask = torch.ones(3, 12, dtype=torch.bool)
    output = reasoner(source, mask, return_trajectory=True)
    assert output["logits"].shape == (3, 10)
    assert output["final_state"].shape == (3, 4, 32)
    assert len(output["trajectory"]) == 2

    no_latent = reasoner(source, mask, intervention="zero")
    shuffled = reasoner(source, mask, intervention="batch_shuffle", intervention_step=0)
    truncated = reasoner(source, mask, truncate_steps=1)
    assert no_latent["logits"].shape == shuffled["logits"].shape == truncated["logits"].shape
    assert assert_no_answer_bypass(reasoner)["passed"]

    mean_initialized = LatentReasoner(
        backbone,
        LatentConfig(
            latent_tokens=4,
            recurrent_steps=2,
            recurrent_blocks=2,
            cross_attention_heads=4,
            source_mean_init=True,
            transition_gate_init=-5.0,
        ),
    )
    mean_output = mean_initialized(source, mask)
    assert mean_output["logits"].shape == (3, 10)
    assert assert_no_answer_bypass(mean_initialized)["passed"]

    cross_only = LatentReasoner(
        backbone,
        LatentConfig(
            latent_tokens=4,
            recurrent_steps=2,
            recurrent_blocks=0,
            cross_attention_heads=4,
        ),
    )
    cross_only_output = cross_only(source, mask)
    assert cross_only_output["logits"].shape == (3, 10)
    assert cross_only.recurrent_layer_indices == ()

    adapted = LatentReasoner(
        backbone,
        LatentConfig(
            latent_tokens=4,
            recurrent_steps=2,
            recurrent_blocks=0,
            cross_attention_heads=4,
            source_adapter_backend="mlp",
            source_adapter_bottleneck=8,
        ),
    )
    adapted_output = adapted(source, mask)
    assert adapted_output["logits"].shape == (3, 10)
    assert assert_no_answer_bypass(adapted)["passed"]

    mlp_reasoner = LatentReasoner(
        backbone,
        LatentConfig(
            latent_tokens=4,
            recurrent_steps=2,
            recurrent_blocks=2,
            cross_attention_heads=4,
            source_mean_init=True,
            transition_backend="mlp",
            transition_bottleneck=8,
        ),
    )
    mlp_output = mlp_reasoner(source, mask, return_trajectory=True)
    assert mlp_output["logits"].shape == (3, 10)
    assert len(mlp_output["trajectory"]) == 2
    assert mlp_reasoner.recurrent_layer_indices == ()

    flattened = LatentReasoner(
        backbone,
        LatentConfig(
            latent_tokens=4,
            recurrent_steps=2,
            recurrent_blocks=0,
            cross_attention_heads=4,
            answer_pooling="flatten",
        ),
    )
    flattened_output = flattened(source, mask)
    assert flattened_output["logits"].shape == (3, 10)
    assert assert_no_answer_bypass(flattened)["answer_head_input"] == "final latent tokens flattened"

    reread = LatentReasoner(
        backbone,
        LatentConfig(
            latent_tokens=4,
            recurrent_steps=2,
            recurrent_blocks=0,
            cross_attention_heads=4,
            source_read_each_step=True,
        ),
    )
    reread_output = reread(source, mask, return_trajectory=True)
    assert reread_output["logits"].shape == (3, 10)
    assert len(reread_output["trajectory"]) == 2
