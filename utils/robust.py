import torch

def mad_zscores(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Median‑absolute‑deviation z‑score (robust outlier metric).
    Returns z for each element of `x`.
    """
    med = x.median()
    mad = (x - med).abs().median() + eps
    return 0.6745 * (x - med).abs() / mad
