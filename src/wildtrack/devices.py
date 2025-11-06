import torch

def pick_device(prefer: str = "auto") -> str:
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    if prefer in ("auto", "mps"):
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    return "cpu"

def is_mps(device: str) -> bool:
    return device == "mps" and torch.backends.mps.is_available()

def is_cuda(device: str) -> bool:
    return device.startswith("cuda") and torch.cuda.is_available()

class maybe_autocast:
    """Context manager that autocasts on MPS/CUDA float16, else no-op."""
    def __init__(self, device: str):
        self.device = device
        self.ctx = None
    def __enter__(self):
        if is_mps(self.device):
            torch.set_float32_matmul_precision('high')
            self.ctx = torch.autocast(device_type="mps", dtype=torch.float16)
            return self.ctx.__enter__()
        if is_cuda(self.device):
            self.ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
            return self.ctx.__enter__()
        return None
    def __exit__(self, exc_type, exc, tb):
        if self.ctx is not None:
            return self.ctx.__exit__(exc_type, exc, tb)
        return False
