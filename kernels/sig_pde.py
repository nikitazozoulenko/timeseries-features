from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import numpy as np
import torch
from torch import Tensor
import sigkernel
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kernels.static_kernels import StaticKernel, AbstractKernel, RBFKernel


class CrisStaticWrapper:
    def __init__(
            self, 
            kernel: StaticKernel,
        ):
        """Wrapper for static kernels for Cris Salvi's sigkernel library"""
        self.kernel = kernel


    def batch_kernel(
            self, 
            X:Tensor, 
            Y:Tensor
        ) -> Tensor:
        """
        Outputs k(X^i_t, Y^j_t)

        Args:
            X (Tensor): Tensor of shape (N, T1, d)
            Y (Tensor): Tensor of shape (N, T2, d)

        Returns:
            Tensor: Tensor of shape (N, T1, T2)
        """
        X = X.transpose(1,0)
        Y = Y.transpose(1,0)
        trans_gram = self.kernel.gram(X, Y) # shape (T1, T2, N)
        return trans_gram.permute(2, 0, 1)


    def Gram_matrix(
            self, 
            X: Tensor, 
            Y: Tensor
        ) -> Tensor:
        """
        Outputs k(X^i_s, Y^j_t)
        
        Args:
            X (Tensor): Tensor of shape (N1, T1, d)
            Y (Tensor): Tensor of shape (N2, T2, d)
        
        Returns:
            Tensor: Tensor of shape (N1, N2, T1, T2)
        """
        N1, T1, d = X.shape
        N2, T2, d = Y.shape
        X = X.reshape(-1, d)
        Y = Y.reshape(-1, d)
        flat_gram = self.kernel.gram(X, Y) # shape (N1 * T1, N2 * T2)
        gram = flat_gram.reshape(N1, T1, N2, T2)
        return gram.permute(0, 2, 1, 3)
    
    

class SigPDEKernel(AbstractKernel):
    def __init__(
            self,
            static_kernel: StaticKernel = RBFKernel(),
            dyadic_order:int = 1,
            max_batch:int = 10,
        ):
        """
        Signature PDE kernel for timeseries (x_1, ..., x_T) in R^d,
        kernelized with a static kernel k : R^d x R^d -> R.

        Args:
            static_kernel (StaticKernel): Static kernel on R^d.
            dyadic_order (int, optional): Dyadic order in PDE solver. Defaults to 1.
            max_batch (int, optional): Max batch size for computations. Defaults to 10.
        """
        self.static_wrapper = CrisStaticWrapper(static_kernel)
        self.dyadic_order = dyadic_order
        self.sig_ker = sigkernel.SigKernel(self.static_wrapper, dyadic_order)
        self.max_batch = max_batch


    def gram(
            self, 
            X: Tensor, 
            Y: Tensor, 
            diag: bool = False, 
        ):
        """
        Computes the Gram matrix K(X_i, Y_j), or the diagonal K(X_i, Y_i) 
        if diag=True. The time series in X are of shape (T1, d), and the
        time series in Y are of shape (T2, d), where d is the path dimension.

        Args:
            X (Tensor): Tensor with shape (N1, T1, d).
            Y (Tensor): Tensor with shape (N2, T2, d).
            diag (bool, optional): If True, only computes the kernel for the 
                pairs K(X_i, Y_i). Defaults to False.

        Returns:
            Tensor: Tensor with shape (N1, N2), or (N1) if diag=True.
        """
        if diag:
            return self.sig_ker.compute_kernel(X, Y, self.max_batch)
        else:
            return self.sig_ker.compute_Gram(X, Y, sym=(X is Y), max_batch=self.max_batch)


    def __call__(
            self, 
            X: Tensor, 
            Y: Tensor, 
        )->Tensor:
        """
        Computes the kernel evaluation k(X, Y) of two time series 
        (with batch support). The time series in X are of shape (T1, d), 
        and the time series in Y are of shape (T2, d), where d is the 
        path dimension.

        Args:
            X (Tensor): Tensor with shape (... , T1, d).
            Y (Tensor): Tensor with shape (... , T2, d), with (...) same as X.
        
        Returns:
            Tensor: Tensor with shape (...).
        """
        if X.ndim == 2 and Y.ndim == 2:
            X = X.unsqueeze(0)
            Y = Y.unsqueeze(0)
        return self.sig_ker.compute_kernel(X, Y, self.max_batch)

