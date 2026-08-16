"""Tests for the model registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from vibeforge.router.registry import ConfigError, ModelRegistry, find_models_file
from vibeforge.types import Complexity, ModelTier

DEFAULT_YAML = """\
models:
  - name: tiny-fast
    ollama_tag: qwen2.5:0.5b
    complexity_ceiling: trivial
    approx_ram_gb: 0.6
    notes: fastest
  - name: balanced
    ollama_tag: llama3.1:latest
    complexity_ceiling: medium
    approx_ram_gb: 4.9
    notes: everyday
  - name: heavy
    ollama_tag: qwen2.5-coder:14b
    complexity_ceiling: high
    approx_ram_gb: 9.0
    notes: strongest
"""


@pytest.fixture
def default_registry() -> ModelRegistry:
    """The standard three-tier registry used across tests."""
    return ModelRegistry.from_yaml(DEFAULT_YAML)


def write_yaml(tmp_path: Path, content: str) -> Path:
    """Write a YAML config file into a temp dir and return its path."""
    path = tmp_path / "models.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_all_tiers(default_registry: ModelRegistry) -> None:
    names = [model.name for model in default_registry.models]
    assert names == ["tiny-fast", "balanced", "heavy"]


def test_models_are_sorted_cheapest_first(default_registry: ModelRegistry) -> None:
    rams = [model.approx_ram_gb for model in default_registry.models]
    assert rams == sorted(rams)


def test_pick_trivial_goes_to_tiny(default_registry: ModelRegistry) -> None:
    assert default_registry.pick_for(Complexity.TRIVIAL).name == "tiny-fast"


def test_pick_low_goes_to_balanced(default_registry: ModelRegistry) -> None:
    assert default_registry.pick_for(Complexity.LOW).name == "balanced"


def test_pick_medium_goes_to_balanced(default_registry: ModelRegistry) -> None:
    assert default_registry.pick_for(Complexity.MEDIUM).name == "balanced"


def test_pick_high_goes_to_heavy(default_registry: ModelRegistry) -> None:
    assert default_registry.pick_for(Complexity.HIGH).name == "heavy"


def test_tier_at_ceiling_is_covered(default_registry: ModelRegistry) -> None:
    assert default_registry.pick_for(Complexity.MEDIUM).name == "balanced"
    assert default_registry.pick_for(Complexity.TRIVIAL).name == "tiny-fast"


def test_cheapest_covering_model_wins() -> None:
    registry = ModelRegistry.from_yaml("""\
        models:
          - name: big
            ollama_tag: big-model:latest
            complexity_ceiling: high
            approx_ram_gb: 40
          - name: mid
            ollama_tag: mid-model:latest
            complexity_ceiling: high
            approx_ram_gb: 16
        """)
    assert registry.pick_for(Complexity.HIGH).name == "mid"


def test_falls_back_to_most_capable_above_max_ceiling() -> None:
    registry = ModelRegistry.from_yaml("""\
        models:
          - name: weak
            ollama_tag: weak:latest
            complexity_ceiling: trivial
            approx_ram_gb: 1
          - name: middle
            ollama_tag: middle:latest
            complexity_ceiling: medium
            approx_ram_gb: 8
        """)
    assert registry.pick_for(Complexity.HIGH).name == "middle"


def test_falls_back_with_single_model() -> None:
    registry = ModelRegistry.from_yaml("""\
        models:
          - name: only
            ollama_tag: only:latest
            complexity_ceiling: trivial
            approx_ram_gb: 1
        """)
    assert registry.pick_for(Complexity.HIGH).name == "only"


def test_loads_from_file(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, DEFAULT_YAML)
    registry = ModelRegistry.from_yaml_file(path)
    assert len(registry.models) == 3


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        ModelRegistry.from_yaml_file(tmp_path / "does-not-exist.yaml")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, "models: [unclosed")
    with pytest.raises(ConfigError):
        ModelRegistry.from_yaml_file(path)


def test_missing_models_key_raises() -> None:
    with pytest.raises(ConfigError):
        ModelRegistry.from_yaml("tiers: []")


def test_empty_models_list_raises() -> None:
    with pytest.raises(ConfigError):
        ModelRegistry.from_yaml("models: []")


def test_missing_required_field_raises() -> None:
    with pytest.raises(ConfigError):
        ModelRegistry.from_yaml("""\
            models:
              - name: partial
                ollama_tag: partial:latest
            """)


def test_invalid_ceiling_raises() -> None:
    with pytest.raises(ConfigError):
        ModelRegistry.from_yaml("""\
            models:
              - name: bad
                ollama_tag: bad:latest
                complexity_ceiling: extreme
                approx_ram_gb: 1
            """)


def test_nonpositive_ram_raises() -> None:
    with pytest.raises(ConfigError):
        ModelRegistry.from_yaml("""\
            models:
              - name: bad
                ollama_tag: bad:latest
                complexity_ceiling: high
                approx_ram_gb: 0
            """)


def test_duplicate_tier_names_raise() -> None:
    with pytest.raises(ConfigError):
        ModelRegistry.from_yaml("""\
            models:
              - name: dup
                ollama_tag: a:latest
                complexity_ceiling: high
                approx_ram_gb: 1
              - name: dup
                ollama_tag: b:latest
                complexity_ceiling: high
                approx_ram_gb: 2
            """)


def test_bundled_default_config_is_valid() -> None:
    registry = ModelRegistry.load_default()
    assert len(registry.models) >= 3
    for model in registry.models:
        assert isinstance(model, ModelTier)
        assert model.approx_ram_gb > 0


def test_unexpected_tier_key_is_rejected() -> None:
    with pytest.raises(ConfigError):
        ModelRegistry.from_yaml("""\
            models:
              - name: bad
                ollama_tag: bad:latest
                complexity_ceiling: high
                approx_ram_gb: 1
                complicancy_ceiling: typo
            """)


def test_unexpected_top_level_key_is_rejected() -> None:
    with pytest.raises(ConfigError):
        ModelRegistry.from_yaml("""\
            models:
              - name: ok
                ollama_tag: ok:latest
                complexity_ceiling: high
                approx_ram_gb: 1
            extra_stuff: true
            """)


def test_nan_ram_is_rejected() -> None:
    with pytest.raises(ConfigError):
        ModelRegistry.from_yaml("""\
            models:
              - name: bad
                ollama_tag: bad:latest
                complexity_ceiling: high
                approx_ram_gb: nan
            """)


def test_infinite_ram_is_rejected() -> None:
    with pytest.raises(ConfigError):
        ModelRegistry.from_yaml("""\
            models:
              - name: bad
                ollama_tag: bad:latest
                complexity_ceiling: high
                approx_ram_gb: inf
            """)


def test_empty_ollama_tag_is_rejected() -> None:
    with pytest.raises(ConfigError):
        ModelRegistry.from_yaml("""\
            models:
              - name: bad
                ollama_tag: ""
                complexity_ceiling: high
                approx_ram_gb: 1
            """)


def test_wrong_notes_type_is_rejected() -> None:
    with pytest.raises(ConfigError):
        ModelRegistry.from_yaml("""\
            models:
              - name: bad
                ollama_tag: bad:latest
                complexity_ceiling: high
                approx_ram_gb: 1
                notes: [not, a, string]
            """)


def test_top_level_list_is_rejected() -> None:
    with pytest.raises(ConfigError):
        ModelRegistry.from_yaml("- not\n- a\n- mapping\n")


def test_error_message_names_exact_location() -> None:
    with pytest.raises(ConfigError) as excinfo:
        ModelRegistry.from_yaml("""\
            models:
              - name: ok
                ollama_tag: ok:latest
                complexity_ceiling: high
                approx_ram_gb: 1
              - name: bad
                ollama_tag: bad:latest
                complexity_ceiling: medium
                approx_ram_gb: 0
            """)
    message = str(excinfo.value)
    assert "models[1].approx_ram_gb" in message
    assert "greater than 0" in message


def test_pick_is_deterministic_on_ram_tie() -> None:
    registry = ModelRegistry.from_yaml("""\
        models:
          - name: beta
            ollama_tag: beta:latest
            complexity_ceiling: high
            approx_ram_gb: 8
          - name: alpha
            ollama_tag: alpha:latest
            complexity_ceiling: high
            approx_ram_gb: 8
        """)
    assert registry.pick_for(Complexity.HIGH).name == "alpha"


def test_find_models_file_prefers_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VIBEFORGE_MODELS", raising=False)
    monkeypatch.chdir(tmp_path)
    write_yaml(tmp_path, DEFAULT_YAML)
    assert find_models_file() == tmp_path / "models.yaml"


def test_find_models_file_falls_back_to_bundled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("VIBEFORGE_MODELS", raising=False)
    monkeypatch.chdir(tmp_path)
    result = find_models_file()
    assert result.is_file()
    assert "data" in result.parts


def test_find_models_file_honors_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = write_yaml(tmp_path, DEFAULT_YAML)
    monkeypatch.setenv("VIBEFORGE_MODELS", str(path))
    assert find_models_file() == path


def test_find_models_file_env_pointing_nowhere_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VIBEFORGE_MODELS", str(tmp_path / "nope.yaml"))
    with pytest.raises(ConfigError):
        find_models_file()
