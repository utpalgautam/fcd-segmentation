"""Optional PyTorch device selection."""

def get_device(requested: str = "auto"):
    """Choose auto (CUDA when available, otherwise CPU), cpu, or cuda."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for this function. Install a compatible build from "
            "https://pytorch.org/get-started/locally/."
        ) from exc
    requested = requested.strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable to this PyTorch install.")
        return torch.device("cuda")
    raise ValueError("requested must be one of: auto, cpu, cuda")
