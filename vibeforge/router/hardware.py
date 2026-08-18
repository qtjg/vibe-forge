"""Hardware probing for tier calibration: RAM, VRAM, and ``ollama list``.

Used by ``vibeforge doctor`` to cross-reference the machine ("is this tier
going to fit?") and the local Ollama ("is this pulled model configured?").
Everything here is read-only and every probe is injectable:

- psutil (optional, behind the ``[hardware]`` extra) for system RAM —
  when it is missing, RAM stays ``None`` instead of crashing.
- ``nvidia-smi`` on PATH for GPU VRAM — absent tooling means VRAM stays
  ``None``.
- ``ollama list`` for pulled models, parsed from plain CLI output.

Subprocess and psutil calls are never hardwired: callers inject a
``subprocess_runner`` / ``psutil_module``, so tests run with no real
hardware, no Ollama, and no psutil dependency.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = [
    "HardwareInfo",
    "detect",
    "available_ram_gb",
    "nvidia_vram_gb",
    "pulled_ollama_models",
    "guess_ram_gb",
    "guess_ceiling",
]

#: Rough RAM-per-model-size heuristic: quantized size on disk, plus a bit
#: of breathing room for context and KV cache.
_SIZE_HEADROOM = 1.15

#: Fallback RAM guess by parametrization in the tag (e.g. ``qwen2.5:3b``),
#: used when the size column is unavailable. Rough, statement-grade only.
_PARAM_RAM_GUESSES: dict[float, float] = {
    0.5: 0.6,
    1.0: 1.3,
    1.5: 1.5,
    3.0: 2.4,
    4.0: 4.9,
    7.0: 4.9,
    8.0: 6.1,
    13.0: 9.0,
    14.0: 9.0,
    32.0: 18.0,
    70.0: 40.0,
}


@dataclass(frozen=True)
class HardwareInfo:
    """What the machine reports, or ``None`` when a probe is unavailable.

    Attributes:
        total_ram_gb: System RAM in GiB (psutil), or ``None``.
        total_vram_gb: Largest single-GPU VRAM in GiB (nvidia-smi),
            or ``None`` when nvidia-smi is absent or unreadable.
        ram_detected: True when psutil was importable and readable.
        vram_detected: True when nvidia-smi was found and parsed.
    """

    total_ram_gb: float | None = None
    total_vram_gb: float | None = None
    ram_detected: bool = False
    vram_detected: bool = False

    @property
    def missing_psutil(self) -> bool:
        """True when psutil is not installed (advise the hardware extra)."""
        return not self.ram_detected


#: A ``subprocess.run``-shaped callable; injected in tests.
SubprocessRunner = Callable[[Sequence[str]], object]


def _default_runner(cmd: Sequence[str]) -> object:
    """Run ``cmd`` with output captured, bounded by a sane timeout."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30)


def _quietly(fn: Callable[[], object]) -> object | None:
    """Run ``fn``; any exception means "no data" for that probe."""
    try:
        return fn()
    except Exception:  # noqa: BLE001 -- probes must never crash the doctor
        return None


def available_ram_gb(psutil_module: object | None = None) -> float | None:
    """Total system RAM in GiB, or ``None`` when psutil is unavailable.

    Args:
        psutil_module: The psutil module to read (``None`` imports it
            lazily; tests inject a stub).
    """
    if psutil_module is None:
        try:
            import psutil as psutil_module
        except ImportError:
            return None
    total = _quietly(lambda: psutil_module.virtual_memory().total)
    if not isinstance(total, (int, float)) or total <= 0:
        return None
    return round(total / (2**30), 2)


def nvidia_vram_gb(subprocess_runner: SubprocessRunner | None = None) -> float | None:
    """Largest single-GPU VRAM in GiB, or ``None`` without nvidia-smi.

    VRAM must fit one GPU, so the largest card is the binding limit.

    Args:
        subprocess_runner: Injected runner (``None`` uses ``subprocess.run``
            with a timeout).
    """
    if shutil.which("nvidia-smi") is None:
        return None
    runner = subprocess_runner or _default_runner
    result = _quietly(
        lambda: runner(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    )
    if result is None:
        return None
    stdout = getattr(result, "stdout", "")
    if getattr(result, "returncode", 0) != 0 or not stdout:
        return None
    sizes = []
    for line in stdout.splitlines():
        text = line.strip().rstrip("MiB").strip()
        if text.isdigit():
            sizes.append(int(text))
    if not sizes:
        return None
    return round(max(sizes) / 1024.0, 2)


def pulled_ollama_models(subprocess_runner: SubprocessRunner | None = None) -> set[str]:
    """Tags reported by ``ollama list``, or an empty set on any failure.

    Parser is lenient on purpose: the header row (``NAME``) is skipped and
    the first column of every other row is the tag.

    Args:
        subprocess_runner: Injected runner (``None`` shells out to
            ``ollama list`` directly).
    """
    runner = subprocess_runner or _default_runner
    result = _quietly(lambda: runner(["ollama", "list"]))
    if result is None or getattr(result, "returncode", 0) != 0:
        return set()
    stdout = getattr(result, "stdout", "") or ""
    tags = set()
    for line in stdout.splitlines():
        columns = line.split()
        if len(columns) < 2 or columns[0] == "NAME":
            continue
        tags.add(columns[0])
    return tags


def detect(
    psutil_module: object | None = None,
    subprocess_runner: SubprocessRunner | None = None,
) -> HardwareInfo:
    """Probe RAM and VRAM with graceful degradation.

    Args:
        psutil_module: Injected psutil module for RAM (see
            :func:`available_ram_gb`).
        subprocess_runner: Injected subprocess runner (see
            :func:`nvidia_vram_gb`).
    """
    ram = available_ram_gb(psutil_module)
    vram = nvidia_vram_gb(subprocess_runner)
    return HardwareInfo(
        total_ram_gb=ram,
        total_vram_gb=vram,
        ram_detected=ram is not None,
        vram_detected=vram is not None,
    )


def guess_ram_gb(size_bytes: int | None, tag: str) -> float | None:
    """Best-effort RAM guess for a pulled model.

    Prefers the model size reported by ``ollama list`` (size on disk times a
    small headroom factor); falls back to the parametrization embedded in
    the tag (``qwen2.5:3b`` -> ``2.4``). ``None`` when nothing is usable.
    """
    if isinstance(size_bytes, (int, float)) and size_bytes > 0:
        return round(size_bytes / (2**30) * _SIZE_HEADROOM, 2)
    params = tag.split(":")[-1].rstrip("bB") or tag
    try:
        number = float(params)
    except ValueError:
        return None
    return _PARAM_RAM_GUESSES.get(number)


def guess_ceiling(ram_gb: float | None) -> str:
    """A sensible complexity ceiling for the guessed size.

    Ranges are deliberately rough: the doctor's suggestion is a scaffold,
    not a guarantee.
    """
    if ram_gb is None or ram_gb < 0.8:
        return "trivial"
    if ram_gb <= 2.0:
        return "low"
    if ram_gb <= 6.0:
        return "medium"
    return "high"
