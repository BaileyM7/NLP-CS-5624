"""Model registry: the short names used across the project, mapped to OpenRouter model IDs."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str  # short name used in run IDs, file names and CLI flags
    openrouter_id: str
    main_run: bool  # one of the four models in the main experiment


_SPECS = [
    ModelSpec("opus-5.5", "anthropic/claude-opus-5.5", main_run=True),
    ModelSpec("haiku-5.5", "anthropic/claude-haiku-5.5", main_run=True),
    ModelSpec("gpt-6-astra", "openai/gpt-6-astra", main_run=True),
    ModelSpec("gpt-6-luna", "openai/gpt-6-luna", main_run=True),
    ModelSpec("sonnet-5.5", "anthropic/claude-sonnet-5.5", main_run=False),  # writes the silently_wrong fact edits
    ModelSpec("gpt-6.1-sol", "openai/gpt-6.1-sol", main_run=False),
]

MODELS: dict[str, ModelSpec] = {spec.name: spec for spec in _SPECS}
MAIN_RUN_MODELS: list[str] = [spec.name for spec in _SPECS if spec.main_run]

# Fixed roles from docs/experiment-spec.md; the verifier and judge must be different models
VERIFIER_MODEL = "gpt-6-astra"
JUDGE_MODEL = "opus-5.5"
FACT_EDIT_MODEL = "sonnet-5.5"


def get_model(name_or_id: str) -> ModelSpec:
    """Look up a model by short name ("opus-5.5") or OpenRouter ID ("anthropic/claude-opus-5.5")."""
    if name_or_id in MODELS:
        return MODELS[name_or_id]
    for spec in _SPECS:
        if spec.openrouter_id == name_or_id:
            return spec
    raise ValueError(f"unknown model {name_or_id!r}; known models: {', '.join(MODELS)}")
