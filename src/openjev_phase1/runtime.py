"""Hardware selection and synchronization policy for scorer runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class RuntimeSpec:
    requested_device: str
    device: str
    dtype: str


def _torch(torch_module=None):
    return torch_module if torch_module is not None else import_module("torch")


def resolve_runtime(requested_device: str, torch_module=None) -> RuntimeSpec:
    torch = _torch(torch_module)
    if requested_device not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unknown device {requested_device!r}")

    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0
    if requested_device in {"auto", "cuda"} and cuda_available:
        if cuda_count != 1:
            raise ValueError("Expose exactly one CUDA GPU, for example with CUDA_VISIBLE_DEVICES")
        return RuntimeSpec(requested_device, "cuda:0", "bfloat16")
    if requested_device == "cuda":
        raise ValueError("CUDA was requested but is unavailable")

    mps_available = torch.backends.mps.is_available()
    if requested_device in {"auto", "mps"} and mps_available:
        return RuntimeSpec(requested_device, "mps", "float16")
    if requested_device == "mps":
        raise ValueError("MPS was requested but is unavailable")
    return RuntimeSpec(requested_device, "cpu", "float32")


def validate_mode_device(mode: str, device: str) -> None:
    if device != "cuda:0" and mode != "direct":
        raise ValueError(f"Device {device} currently supports only direct mode")


def synchronize(device, torch_module=None) -> None:
    torch = _torch(torch_module)
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    elif device_type == "mps":
        torch.mps.synchronize()
