from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import numpy as np
import torch
from torch import Tensor
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kernels.static_kernels import StaticKernel, AbstractKernel
from kernels.static_kernels import RBFKernel

#######################################################################################
################### Time series Integral Kernel of static kernel ######################
#######################################################################################

class IntegralKernel(AbstractKernel):
    def __init__(
            self,
            static_kernel:StaticKernel = RBFKernel,
        ):
        """
        The integral kernel K(x, y) = \int k(x_t, y_t) dt, given a static kernel 
        k(x, y) on R^d.

        Args:
            static_kernel (StaticKernel): Static kernel on R^d.
        """
        super().__init__()
        self.static_kernel = static_kernel


    def gram(
            self, 
            X: Tensor, 
            Y: Tensor, 
            diag: bool = False, 
        ):
        """
        Computes the Gram matrix K(X_i, Y_j), or the diagonal K(X_i, Y_i) 
        if diag=True. The time series in X and Y are assumed to be of shape (T, d).

        Args:
            X (Tensor): Tensor with shape (N1, ..., T, d).
            Y (Tensor): Tensor with shape (N2, ..., T, d), with (...) same as X.
            diag (bool, optional): If True, only computes the kernel for the 
                pairs K(X_i, Y_i). Defaults to False.

        Returns:
            Tensor: Tensor with shape (N1, N2, ...), or (N1, ...) if diag=True.
        """
        #Shape (N1, N2, ..., T), or (N1, ..., T) if diag=True
        ijKt = self.static_kernel.gram(X, Y, diag)

        #return integral of k(x_t, y_t) dt for each pair x and y
        T = X.shape[-2]
        return torch.trapz(ijKt, dx=1/(T-1), axis=-1)


    def __call__(
            self, 
            X: Tensor, 
            Y: Tensor, 
        )->Tensor:
        """
        Computes the kernel evaluation k(X, Y) of two time series
        (with batch support).

        Args:
            X (Tensor): Tensor with shape (... , T, d).
            Y (Tensor): Tensor with shape (... , T, d), with (...) same as X.
        
        Returns:
            Tensor: Tensor with shape (...).
        """
        if X.ndim==2 and Y.ndim==2:
            X = X.unsqueeze(0)
            Y = Y.unsqueeze(0)
        return self.gram(X, Y, diag=True)
