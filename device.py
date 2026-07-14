"""Device auto-detection: CUDA > MPS (Apple Silicon) > CPU.

Nothing in this project previously called this -- every device= argument was
hardcoded to "cpu" or derived from an already-CPU tensor, so the project
silently never used a GPU even when one was available. Import get_device()
and call ensemble.to(device) / actor_critic.to(device) once after
construction (or after loading a checkpoint); training loops and inference
call sites derive their device from the model's own parameters, so nothing
else needs to change."""
import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
