import pytest

from openjev_phase1.runtime import resolve_runtime, synchronize, validate_mode_device


class FakeCuda:
    def __init__(self, available=False, count=0):
        self.available = available
        self.count = count
        self.synchronized = []

    def is_available(self):
        return self.available

    def device_count(self):
        return self.count

    def synchronize(self, device):
        self.synchronized.append(device)


class FakeMpsBackend:
    def __init__(self, available=False):
        self.available = available

    def is_available(self):
        return self.available


class FakeMps:
    def __init__(self):
        self.calls = 0

    def synchronize(self):
        self.calls += 1


class FakeTorch:
    def __init__(self, *, cuda=False, cuda_count=0, mps=False):
        self.cuda = FakeCuda(cuda, cuda_count)
        self.backends = type("Backends", (), {"mps": FakeMpsBackend(mps)})()
        self.mps = FakeMps()


@pytest.mark.parametrize(
    ("torch_module", "expected_device", "expected_dtype"),
    [
        (FakeTorch(cuda=True, cuda_count=1, mps=True), "cuda:0", "bfloat16"),
        (FakeTorch(mps=True), "mps", "float16"),
        (FakeTorch(), "cpu", "float32"),
    ],
)
def test_auto_runtime_precedence(torch_module, expected_device, expected_dtype):
    runtime = resolve_runtime("auto", torch_module)
    assert runtime.requested_device == "auto"
    assert runtime.device == expected_device
    assert runtime.dtype == expected_dtype


def test_auto_rejects_multiple_visible_cuda_devices():
    with pytest.raises(ValueError, match="exactly one CUDA GPU"):
        resolve_runtime("auto", FakeTorch(cuda=True, cuda_count=2, mps=True))


@pytest.mark.parametrize("requested", ["cuda", "mps"])
def test_explicit_unavailable_accelerator_is_rejected(requested):
    with pytest.raises(ValueError, match=requested.upper()):
        resolve_runtime(requested, FakeTorch())


@pytest.mark.parametrize("device", ["mps", "cpu"])
@pytest.mark.parametrize("mode", ["serial", "shared", "reranker"])
def test_non_cuda_runtime_only_accepts_direct(mode, device):
    with pytest.raises(ValueError, match="direct"):
        validate_mode_device(mode, device)


@pytest.mark.parametrize("device", ["cuda:0", "mps", "cpu"])
def test_direct_accepts_every_resolved_device(device):
    validate_mode_device("direct", device)


def test_synchronize_dispatches_to_cuda_and_mps():
    torch_module = FakeTorch()
    synchronize("cuda:0", torch_module)
    synchronize("mps", torch_module)
    synchronize("cpu", torch_module)
    assert torch_module.cuda.synchronized == ["cuda:0"]
    assert torch_module.mps.calls == 1
