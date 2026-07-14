import torch
from device import get_device


def test_get_device_returns_a_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)


def test_get_device_prefers_accelerator_over_cpu_when_available():
    device = get_device()
    if torch.cuda.is_available():
        assert device.type == "cuda"
    elif torch.backends.mps.is_available():
        assert device.type == "mps"
    else:
        assert device.type == "cpu"
