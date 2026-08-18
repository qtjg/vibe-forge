"""Tests for hardware probing and the doctor's hardware-aware findings.

Every probe is mocked: no psutil, no nvidia-smi, no Ollama, and no real
hardware anywhere. The ollama SDK client is stubbed the same way
``test_executor.py`` stubs HTTP: plain object fakes.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from vibeforge.doctor import OK, WARN, Doctor
from vibeforge.router.hardware import (
    HardwareInfo,
    available_ram_gb,
    detect,
    guess_ceiling,
    guess_ram_gb,
    nvidia_vram_gb,
    pulled_ollama_models,
)
from vibeforge.router.registry import ModelRegistry

MODELS = """\
models:
  - name: tiny-fast
    ollama_tag: qwen2.5:0.5b
    complexity_ceiling: high
    approx_ram_gb: 0.6
  - name: heavy
    ollama_tag: qwen2.5-coder:14b
    complexity_ceiling: high
    approx_ram_gb: 9.0
"""


def fake_psutil(total_bytes: int) -> object:
    """A stub psutil module exposing only virtual_memory().total."""

    return SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=total_bytes))


def fake_runner(calls: list[list[str]], returncode: int = 0, stdout: str = "") -> object:
    """A subprocess runner capturing the command and returning fixed output."""

    def run(cmd: list[str]) -> object:
        calls.append(cmd)
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    return run


def make_client(*models: SimpleNamespace) -> object:
    return SimpleNamespace(list=lambda: SimpleNamespace(models=list(models)))


def model(tag: str, size: int = 0) -> SimpleNamespace:
    return SimpleNamespace(model=tag, size=size or 0)


# ---------------------------------------------------------------------------
# hardware.py unit probes
# ---------------------------------------------------------------------------


def test_available_ram_gb_reads_total_rounded_to_gi_b() -> None:
    assert available_ram_gb(fake_psutil(16 * 2**30)) == 16.0
    assert available_ram_gb(fake_psutil(int(8.5 * 2**30))) == pytest.approx(8.5)


def _hide_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    # None in sys.modules makes `import psutil` raise ImportError,
    # regardless of whether the dev venv actually has psutil installed.
    monkeypatch.setitem(sys.modules, "psutil", None)


def test_available_ram_gb_returns_none_without_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    _hide_psutil(monkeypatch)
    assert available_ram_gb(None) is None  # psutil is not a core dependency


def test_nvidia_vram_gb_uses_largest_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vibeforge.router.hardware.shutil.which", lambda _: "/usr/bin/nvidia-smi")
    vram = nvidia_vram_gb(fake_runner([], stdout="8192\n4096\n"))
    assert vram == 8.0


def test_nvidia_vram_gb_none_without_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vibeforge.router.hardware.shutil.which", lambda _: None)
    assert nvidia_vram_gb(fake_runner([])) is None


def test_nvidia_vram_gb_none_on_failure() -> None:
    vram = nvidia_vram_gb(fake_runner([], returncode=1, stdout=""))
    assert vram is None


def test_pulled_ollama_models_skips_header_and_parses_tags() -> None:
    calls: list[list[str]] = []
    tags = pulled_ollama_models(
        fake_runner(
            calls,
            stdout="NAME            ID              SIZE      MODIFIED\n"
            "qwen2.5:0.5b     abc123          473MB     2 days ago\n"
            "llama3.2:1b      def456          1.3 GB    2 days ago\n",
        )
    )
    assert tags == {"qwen2.5:0.5b", "llama3.2:1b"}
    assert calls == [["ollama", "list"]]


def test_pulled_ollama_models_empty_on_failure() -> None:
    assert pulled_ollama_models(fake_runner([], returncode=127, stdout="")) == set()


def test_detect_combines_ram_and_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vibeforge.router.hardware.shutil.which", lambda _: "/usr/bin/nvidia-smi")
    info = detect(
        psutil_module=fake_psutil(16 * 2**30), subprocess_runner=fake_runner([], stdout="8192\n")
    )
    assert info.total_ram_gb == 16.0
    assert info.total_vram_gb == 8.0
    assert info.ram_detected
    assert info.vram_detected


def test_detect_degrades_when_every_probe_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hide_psutil(monkeypatch)
    monkeypatch.setattr("vibeforge.router.hardware.shutil.which", lambda _: None)
    info = detect()
    assert info.total_ram_gb is None
    assert info.total_vram_gb is None
    assert info.missing_psutil


def test_guess_ram_gb_prefers_reported_size() -> None:
    assert guess_ram_gb(int(7 * 2**30), "anything") == pytest.approx(8.05, abs=0.01)
    assert guess_ram_gb(None, "qwen2.5:3b") == 2.4
    assert guess_ram_gb(None, "random-name") is None


def test_guess_ceiling_brackets() -> None:
    assert guess_ceiling(0.5) == "trivial"
    assert guess_ceiling(1.5) == "low"
    assert guess_ceiling(4.0) == "medium"
    assert guess_ceiling(9.0) == "high"
    assert guess_ceiling(None) == "trivial"


# ---------------------------------------------------------------------------
# doctor flag scenarios (all probes injected, nothing real)
# ---------------------------------------------------------------------------


def test_doctor_flags_tier_exceeding_available_ram() -> None:
    finds = Doctor(
        registry=ModelRegistry.from_yaml(MODELS),
        ollama_client_factory=lambda: make_client(
            model("qwen2.5:0.5b"), model("qwen2.5-coder:14b")
        ),
        hardware=HardwareInfo(total_ram_gb=7.8, ram_detected=True),
    ).run()

    warns = [f for f in finds if f.level == WARN and f.check == "hardware"]
    assert any("'heavy' needs ~9.0 GB" in f.message for f in warns)
    assert not any("'tiny-fast'" in f.message for f in warns)


def test_doctor_flags_tier_exceeding_vram() -> None:
    finds = Doctor(
        registry=ModelRegistry.from_yaml(MODELS),
        ollama_client_factory=lambda: make_client(
            model("qwen2.5:0.5b"), model("qwen2.5-coder:14b")
        ),
        hardware=HardwareInfo(
            total_ram_gb=16.0, ram_detected=True, total_vram_gb=6.0, vram_detected=True
        ),
    ).run()

    warns = [f for f in finds if f.level == WARN and f.check == "hardware"]
    assert any("'heavy' needs ~9.0 GB" in f.message and "GPU VRAM" in f.message for f in warns)


def test_doctor_suggests_tier_for_pulled_model_missing_from_config() -> None:
    sizes = {"llama3.2:1b": int(1.3 * 2**30)}
    finds = Doctor(
        registry=ModelRegistry.from_yaml(MODELS),
        ollama_client_factory=lambda: make_client(
            model("qwen2.5:0.5b"), model("llama3.2:1b", sizes["llama3.2:1b"])
        ),
        hardware=HardwareInfo(total_ram_gb=16.0, ram_detected=True),
    ).run()

    warns = [f for f in finds if f.level == WARN and f.check == "model"]
    hint = [f.message for f in warns if "not in models.yaml" in f.message]
    assert hint
    assert "ollama_tag: llama3.2:1b" in hint[0]
    assert "approx_ram_gb: 1.49" in hint[0]
    assert "complexity_ceiling: low" in hint[0]


def test_doctor_suggests_guess_without_size_column() -> None:
    finds = Doctor(
        registry=ModelRegistry.from_yaml(MODELS),
        ollama_client_factory=lambda: make_client(
            model("qwen2.5:0.5b"), model("qwen2.5-coder:14b", 0), model("llava:7b", 0)
        ),
        hardware=HardwareInfo(total_ram_gb=16.0, ram_detected=True),
    ).run()

    warns = [f for f in finds if f.level == WARN and f.check == "model"]
    hints = [f.message for f in warns if "not in models.yaml" in f.message]
    llava = [h for h in hints if "llava:7b" in h]
    assert any("llava-7b" in h and "approx_ram_gb: 4.9" in h for h in llava)


def test_doctor_warns_when_psutil_missing() -> None:
    finds = Doctor(
        registry=ModelRegistry.from_yaml(MODELS),
        ollama_client_factory=lambda: make_client(
            model("qwen2.5:0.5b"), model("qwen2.5-coder:14b")
        ),
        hardware=HardwareInfo(total_ram_gb=None, ram_detected=False),
    ).run()

    warns = [f for f in finds if f.level == WARN and f.check == "hardware"]
    assert any("vibe-forge[hardware]" in f.message for f in warns)


def test_doctor_skips_vram_without_nvidia_smi() -> None:
    finds = Doctor(
        registry=ModelRegistry.from_yaml(MODELS),
        ollama_client_factory=lambda: make_client(
            model("qwen2.5:0.5b"), model("qwen2.5-coder:14b")
        ),
        hardware=HardwareInfo(total_ram_gb=16.0, ram_detected=True),
    ).run()

    hardware_findings = [f for f in finds if f.check == "hardware"]
    assert any(f.level == OK and "nvidia-smi" in f.message for f in hardware_findings)


def test_doctor_reads_vram_via_detection_when_not_precomputed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("vibeforge.router.hardware.shutil.which", lambda _: "/usr/bin/nvidia-smi")
    finds = Doctor(
        registry=ModelRegistry.from_yaml(MODELS),
        ollama_client_factory=lambda: make_client(
            model("qwen2.5:0.5b"), model("qwen2.5-coder:14b")
        ),
        psutil_module=fake_psutil(32 * 2**30),
        subprocess_runner=fake_runner([], stdout="12288\n"),
    ).run()

    ok = [f for f in finds if f.level == OK and f.check == "hardware"]
    assert any("GPU VRAM: 12.0 GB" in f.message for f in ok)


def test_doctor_never_suggests_configured_tags() -> None:
    finds = Doctor(
        registry=ModelRegistry.from_yaml(MODELS),
        ollama_client_factory=lambda: make_client(
            model("qwen2.5:0.5b"), model("qwen2.5-coder:14b")
        ),
        hardware=HardwareInfo(total_ram_gb=16.0, ram_detected=True),
    ).run()

    assert not any("not in models.yaml" in f.message for f in finds)
