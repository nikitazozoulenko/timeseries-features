from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import torch
from torch import Tensor
import signatory


def sig(
    X: Tensor,
    trunc_level: int,
):
    """
    Computes the truncated signature of time series of
    shape (T,d) with optional batch support.
    
    Args:
        X (Tensor): Tensor of shape (..., T, d) of time series.
        trunc_level (int): Signature truncation level.
    
    Returns:
        Tensor: Tensor of shape (..., D) where 
            D = 1 + d + d^2 + ... + d^trunc_level.
    """
