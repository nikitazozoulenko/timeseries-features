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
        X (Tensor): Tensor of shape (N, T, d) or (T, d) 
            of time series.
        trunc_level (int): Signature truncation level.
    
    Returns:
        Tensor: Tensor of shape (N, D) or (D) where 
            D = 1 + d + d^2 + ... + d^trunc_level.
    """
    if len(X.shape) == 2:
        X = X.unsqueeze(0)
    return signatory.signature(X, trunc_level).squeeze(0)



def logsig(
    X: Tensor,
    trunc_level: int,
):
    """
    Computes the truncated log-signature of time series of
    shape (T,d) with optional batch support.
    
    Args:
        X (Tensor): Tensor of shape (N, T, d) or (T, d) 
            of time series.
        trunc_level (int): Signature truncation level.
    
    Returns:
        Tensor: Tensor of shape (N, D) or (D) where D is 
            O(d^trunc_level) but smaller than the signature.
    """
    if len(X.shape) == 2:
        X = X.unsqueeze(0)
    return signatory.logsignature(X, trunc_level).squeeze(0)
