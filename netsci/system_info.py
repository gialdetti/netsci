#!/usr/bin/env python3
"""
system_info.py — Cross-platform CPU & (compute-capable) GPU identifier

Outputs JSON with:
- cpu: string
- gpus: [all detected GPU/display adapters, best-effort]
- gpus_compute: [subset of gpus that are likely usable for compute (TF/Torch/etc.)]
- os: {system, release, version, machine}

No dependencies required. Optionally uses torch/tensorflow if present.
"""

import json
import platform
import re
import shlex
import subprocess
from shutil import which
from typing import List, Optional


# ---------- utils ----------
def run(cmd: str, timeout: int = 10) -> str:
    try:
        p = subprocess.run(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
            text=True,
        )
        return p.stdout or ""
    except Exception:
        return ""


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in items:
        s = s.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def normalize_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+\(R\)|\s+\(TM\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s)
    return s


# ---------- CPU ----------
def get_cpu_name() -> str:
    # macOS
    if platform.system() == "Darwin":
        out = run("sysctl -n machdep.cpu.brand_string")
        if out.strip():
            return out.strip()

    # Linux
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
        out = run("lscpu")
        m = re.search(r"Model name:\s*(.+)", out)
        if m:
            return m.group(1).strip()

    # Windows
    if platform.system() == "Windows":
        out = run("wmic cpu get Name")
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if len(lines) >= 2:
            return lines[1]
        out = run(
            "powershell -NoProfile -Command Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"
        )
        if out.strip():
            return out.strip()

    # Generic fallback
    return platform.processor() or platform.machine() or "Unknown CPU"


# ---------- GPU (enumeration: try many sources, then dedupe) ----------
def gpus_from_nvidia_smi() -> List[str]:
    if which("nvidia-smi"):
        out = run("nvidia-smi --query-gpu=name --format=csv,noheader")
        return [l.strip() for l in out.splitlines() if l.strip()]
    return []


def gpus_from_windows() -> List[str]:
    if platform.system() != "Windows":
        return []
    out = run("wmic path win32_VideoController get Name")
    items = [l.strip() for l in out.splitlines() if l.strip() and "Name" not in l]
    if items:
        return items
    out = run(
        "powershell -NoProfile -Command Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
    )
    return [l.strip() for l in out.splitlines() if l.strip()]


def gpus_from_macos() -> List[str]:
    if platform.system() != "Darwin":
        return []
    out = run("system_profiler SPDisplaysDataType -detailLevel mini")
    names = []
    for line in out.splitlines():
        line = line.strip()
        m = re.match(r"Chipset Model:\s*(.+)", line)
        if m:
            names.append(m.group(1).strip())
        m2 = re.match(r"(Graphics|GPU):\s*(.+)", line)
        if m2:
            names.append(m2.group(2).strip())
    if not names:
        # Broad vendor hint fallback
        for line in out.splitlines():
            if any(
                v in line
                for v in [
                    "AMD",
                    "Apple",
                    "NVIDIA",
                    "Intel",
                    "Radeon",
                    "GeForce",
                    "Iris",
                    "Arc",
                ]
            ):
                names.append(line.strip())
    return names


def gpus_from_linux_lspci() -> List[str]:
    if platform.system() != "Linux" or not which("lspci"):
        return []
    out = run("lspci -nnk")
    names = []
    for line in out.splitlines():
        if re.search(r"\b(VGA|3D|Display)\b", line, re.IGNORECASE):
            part = line.split(": ", 1)[-1]
            part = re.sub(r"\(rev.*", "", part).strip()
            names.append(part)
    return names


def gpus_from_linux_glxinfo() -> List[str]:
    if platform.system() != "Linux" or not which("glxinfo"):
        return []
    out = run("glxinfo -B")
    m = re.search(r"Device:\s*(.+)", out)
    if m:
        return [m.group(1).strip()]
    m = re.search(r"OpenGL renderer string:\s*(.+)", out)
    if m:
        return [m.group(1).strip()]
    return []


def gpus_from_torch() -> List[str]:
    try:
        import torch  # type: ignore

        names = []
        # CUDA devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    names.append(torch.cuda.get_device_name(i))
                except Exception:
                    pass
        # Apple Metal (MPS)
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Torch doesn't expose a device name; try to map via macOS list later
                names.append("Apple Metal (MPS)")
        except Exception:
            pass
        return names
    except Exception:
        return []


def gpus_from_tensorflow() -> List[str]:
    try:
        import tensorflow as tf  # type: ignore

        devs = tf.config.list_physical_devices("GPU")
        # TF often returns logical names; still useful to assert compute presence
        names = [
            getattr(d, "name", "TensorFlow GPU")
            .replace("/physical_device:", "")
            .strip()
            for d in devs
        ]
        return [n for n in names if n]
    except Exception:
        return []


def get_all_gpu_names() -> List[str]:
    candidates: List[str] = []
    for getter in (
        gpus_from_nvidia_smi,  # NVIDIA
        gpus_from_torch,  # optional runtime libs
        gpus_from_tensorflow,  # optional runtime libs
        gpus_from_windows,  # Windows WMI
        gpus_from_macos,  # macOS system_profiler
        gpus_from_linux_glxinfo,
        gpus_from_linux_lspci,
    ):
        try:
            candidates.extend(getter())
        except Exception:
            pass
    cleaned = [
        normalize_name(c) for c in candidates if isinstance(c, str) and c.strip()
    ]
    return dedupe_keep_order(cleaned)


# ---------- Compute-capable heuristic ----------
def is_compute_capable(name: str) -> bool:
    """
    Heuristic: treat as compute-capable if commonly supported by CUDA/ROCm/Metal/oneAPI.
    This does not guarantee drivers are installed, but matches typical TF/Torch support.
    """
    if not name:
        return False
    s = name.lower()

    # NVIDIA CUDA (most GeForce/Quadro/RTX/GTX)
    if any(
        k in s
        for k in [
            "nvidia",
            "geforce",
            "quadro",
            "rtx",
            "gtx",
            "tesla",
            "l4",
            "a100",
            "h100",
            "h200",
        ]
    ):
        return True

    # AMD ROCm / Radeon / Instinct / Pro (Linux ROCm or macOS Metal)
    if any(k in s for k in ["amd", "radeon", "instinct", "firepro", "radeon pro"]):
        return True

    # Apple Silicon GPUs (Metal via torch.mps / tensorflow-metal)
    if "apple" in s and ("m1" in s or "m2" in s or "m3" in s or "m4" in s):
        return True
    if "apple metal" in s or "mps" in s:
        return True

    # Intel Arc (discrete, oneAPI/Level Zero), often supported for compute
    if "intel arc" in s:
        return True

    # Everything else (e.g., Intel Iris/UHD HD Graphics) => usually NOT compute-capable for TF/Torch
    return False


def refine_compute_with_runtime(gpus_all: List[str]) -> List[str]:
    """
    If torch/tensorflow are present, use their signals to include compute devices even if names are generic.
    """
    runtime_hits = set()

    # Torch CUDA devices (already named above)
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    runtime_hits.add(normalize_name(torch.cuda.get_device_name(i)))
                except Exception:
                    pass
        # Torch MPS
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Try to map an Apple GPU name from gpus_all; else keep generic MPS
                apple_like = [g for g in gpus_all if "apple" in g.lower()]
                runtime_hits.update(apple_like or ["Apple Metal (MPS)"])
        except Exception:
            pass
    except Exception:
        pass

    # TensorFlow GPUs
    try:
        import tensorflow as tf  # type: ignore

        for d in tf.config.list_physical_devices("GPU"):
            # Try to map TF logical GPU to a real name heuristically
            # If we can't, just keep a generic marker to indicate compute presence
            logical = getattr(d, "name", "").lower()
            if "mps" in logical or "metal" in logical:
                apple_like = [g for g in gpus_all if "apple" in g.lower()]
                runtime_hits.update(apple_like or ["Apple Metal (MPS)"])
            else:
                # If we already have detailed names in gpus_all (e.g., NVIDIA), keep them
                for g in gpus_all:
                    if any(
                        k in g.lower()
                        for k in [
                            "nvidia",
                            "geforce",
                            "quadro",
                            "rtx",
                            "gtx",
                            "tesla",
                            "amd",
                            "radeon",
                            "instinct",
                            "intel arc",
                        ]
                    ):
                        runtime_hits.add(g)
                # If nothing matched, at least indicate a generic TF GPU presence
                if not runtime_hits:
                    runtime_hits.add("TensorFlow GPU")
    except Exception:
        pass

    return dedupe_keep_order(list(runtime_hits))


def get_compute_capable_gpu_names(gpus_all: List[str]) -> List[str]:
    # Start with heuristic filter
    heur = [g for g in gpus_all if is_compute_capable(g)]
    # Refine/augment with runtime signals (if torch/tensorflow are present)
    refined = dedupe_keep_order(heur + refine_compute_with_runtime(gpus_all))
    return refined


# ---------- system ----------
def get_system_info() -> None:
    cpu = get_cpu_name()
    gpus_all = get_all_gpu_names()  # may be []
    gpus_compute = get_compute_capable_gpu_names(gpus_all)  # subset (may be [])

    info = {
        "cpu": cpu,
        "gpus": gpus_all,
        "gpus_compute": gpus_compute,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
    }
    # print(json.dumps(info, indent=2, ensure_ascii=False))
    return info


if __name__ == "__main__":
    get_system_info()
