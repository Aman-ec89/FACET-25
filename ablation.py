"""Ablation runners."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict

from model import ModelConfig


def ablation_variants(base: ModelConfig) -> Dict[str, ModelConfig]:
    return {
        "full": base,
        "no_attention": replace(base, use_attention=False),
        "no_subband": replace(base, use_subbands=False),
        "remove_B1": replace(base, remove_b1=True),
        "remove_B3": replace(base, remove_b3=True),
        "tcn_temporal": replace(base, temporal="tcn"),
    }
