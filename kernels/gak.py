from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import torch
from torch import Tensor
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kernels.abstract_base import TimeSeriesKernel, StaticKernel
from kernels.static_kernels import RBFKernel


def sigma_gak(X:Tensor):
    """Computes the recommended sigma parameter for the GAK kernel."""
    pass


@torch.jit.script
def update_antidiag(
        logK:Tensor,
        s:int,
        t:int,
    ):
    """
    Function to be used in the computation of the GAK kernel. 
    """
    if s == 0 and t == 0:
        logK[..., 0, 0] += 0
    elif s == 0:
        logK[..., 0, t] += logK[..., 0, t-1]
    elif t == 0:
        logK[..., s, 0] += logK[..., s-1, 0]
    else:
        logK[..., s, t] += torch.log(
            torch.exp(logK[..., s-1, t-1]) + 
            torch.exp(logK[..., s-1, t  ]) + 
            torch.exp(logK[..., s  , t-1])
        )


@torch.jit.script
def log_global_align(
        K:Tensor, 
    ):
    """
    See fig 2 in 
    https://icml.cc/2011/papers/489_icmlpaper.pdf

    Args:
        K (Tensor): Tensor of shape (..., T1, T2) of Gaussian
            kernel evaluations K(x_s, x_t).
        triangle_param (int): Parameter in the TGAK kernel.
    """
    # make infinitely divisible
    T1, T2 = K.shape[-2:]
    K = K / (2 - K)
    EPS = 1e-10
    logK = torch.log(torch.clamp(K, min=EPS))

    #iterate over antidiagonals
    for diag in range(T1+T2-1):
        futures : List[torch.jit.Future[None]] = []
        for s in range(max(0, diag - T2 + 1), min(diag + 1, T1)):
            t = diag - s
            futures.append( torch.jit.fork(update_antidiag, logK, s, t) )
        [torch.jit.wait(fut) for fut in futures]
    return logK[..., -1, -1]


# My PyTorch implementation is 60x faster compared to tslearn.metrics.cdist_gak,
# but ksig's cuda version is still 10x faster than mine though. JAX is probably faster
@torch.jit.script
def naive_log_global_align(
        K:Tensor, 
    ):
    """
    See fig 2 in 
    https://icml.cc/2011/papers/489_icmlpaper.pdf

    Args:
        K (Tensor): Tensor of shape (..., T1, T2) of Gaussian
            kernel evaluations K(x_s, x_t).
        triangle_param (int): Parameter in the TGAK kernel.
    """
    # make infinitely divisible
    T1, T2 = K.shape[-2:]
    K = K / (2 - K)
    EPS = 1e-10
    logK = torch.log(torch.clamp(K, min=EPS))

    #iterate over antidiagonals
    for s in range(T1):
        for t in range(T2):
            if s == 0 and t == 0:
                logK[..., 0, 0] += 0
            elif s == 0:
                logK[..., 0, t] += logK[..., 0, t-1]
            elif t == 0:
                logK[..., s, 0] += logK[..., s-1, 0]
            else:
                logK[..., s, t] += torch.log(
                    torch.exp(logK[..., s-1, t-1]) + 
                    torch.exp(logK[..., s-1, t  ]) + 
                    torch.exp(logK[..., s  , t-1])
                )
    return logK[..., -1, -1]


class GlobalAlignmentKernel(TimeSeriesKernel):
    def __init__(
            self,
            static_kernel:StaticKernel = RBFKernel(),
            max_batch:int = 500,
            normalize:bool = True,
        ):
        """
        The global alignment kernel of two time series of shape T_i, d), 
        with the respect to a static kernel on R^d. For details see
        https://icml.cc/2011/papers/489_icmlpaper.pdf. Time O(d*T^2) 
        for each pair of time series. Only stable for certain classes
        of static kernels, such as RBF. Note that the static kernel is 
        made into a 'infinitely divisible' kernel through K/(2-K).

        Args:
            static_kernel (StaticKernel): Static kernel on R^d.
        """
        super().__init__(max_batch, normalize)
        self.static_kernel = static_kernel
    

    @property
    def log_space(self):
        return True


    def _batched_ker(
            self, 
            X: Tensor,
            Y: Tensor,
        ):
        """
        Computes the Gram matrix K(X_i, Y_j), or the diagonal K(X_i, Y_i) 
        if diag=True. The time series in X and Y are assumed to be of shape 
        (T1, d) and (T2, d) respectively. O(T^2(d + trunc_level*geo_order^2)) 
        time for each pair of time series.

        Args:
            X (Tensor): Tensor with shape (N, T1, d).
            Y (Tensor): Tensor with shape (N, T2, d).
            diag (bool, optional): If True, only computes the kernel for the 
                pairs K(X_i, Y_i). Defaults to False.

        Returns:
            Tensor: Tensor with shape (N1, N2), or (N1) if diag=True.
        """
        # K shape (N, T1, T2)
        K = self.static_kernel.time_gram(X, Y, diag=True)
        return log_global_align(K)