import torch


def get_torch_device() -> str:
    """Automatically get torch device"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
