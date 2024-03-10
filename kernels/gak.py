from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import torch
from torch import Tensor
import os
import sys
import random

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kernels.abstract_base import TimeSeriesKernel, StaticKernel
from kernels.static_kernels import RBFKernel, LinearKernel

import random

def sigma_gak(
        X:Tensor,
        N_samples:int = 10000,
        seed:Optional[int] = None,
    ):
    """
    Computes the recommended sigma parameter for the GAK kernel,
    i.e. the med(|X^i_s, X^j_t|) * sqrt(T) for a dataset X.

    Args:
        X (Tensor): Tensor of shape (N, T, d).
        N_samples (int): Number of samples to use for the estimation.
        seed (int, optional): Seed for the random number generator. Defaults to None.
    """
    if seed is not None:
        torch.manual_seed(seed)
    N, T, d = X.shape
    N_samples = min(N_samples, N*T)
    indices = random.sample(range(N*T), N_samples)
    X = X.view(-1, d)[indices]

    lin_ker = LinearKernel()
    dists = lin_ker.squared_dist(X, X)
    return torch.sqrt(dists.median()) * T**0.5



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



# My PyTorch implementation is 60x faster compared to tslearn.metrics.cdist_gak (GPU vs GPU),
# but ksig's cuda version is still 10x faster than mine though. JAX is probably even faster
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
        # K shape (N, T1, T2)
        K = self.static_kernel.time_gram(X, Y, diag=True)
        return log_global_align(K)