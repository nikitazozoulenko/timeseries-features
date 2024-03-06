from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import numpy as np
import torch
from torch import Tensor


class AbstractKernel:
    """Abstract kernel class."""

    def gram(self, X: Tensor, Y: Tensor, diag: bool = False):
        """
        Computes the Gram matrix k(X_i, Y_j), or the diagonal k(X_i, Y_i) 
        if diag=True.

        Returns:
            Tensor: Tensor with shape (N1, N2, ...) or (N1, ...) if diag=True.
        """
        raise NotImplementedError("Subclasses must implement gram method.")

    def __call__(self, X: Tensor, Y: Tensor):
        """
        Computes the kernel evaluation k(X, Y) of two tensors (with batch support).
        """
        raise NotImplementedError("Subclasses must implement __call__ method.")


##########################################################################
######################## Static Kernels on R^d ###########################
##########################################################################


class StaticKernel(AbstractKernel):
    """Static kernel k : R^d x R^d -> R."""

    def __init__(self):
        super().__init__()
    

    def gram(
            self, 
            X: Tensor, 
            Y: Tensor, 
            diag: bool = False, 
        )->Tensor:
        """
        Computes the Gram matrix k(X_i, Y_j), or the diagonal k(X_i, Y_i) 
        if diag=True.

        Args:
            X (Tensor): Tensor with shape (N1, ... , d).
            Y (Tensor): Tensor with shape (N2, ... , d).
            diag (bool, optional): If True, only computes the kernel for the 
                pairs k(X_i, Y_i). Defaults to False.

        Returns:
            Tensor: Tensor with shape (N1, N2, ...) or (N1, ...) if diag=True.
        """
        raise NotImplementedError("Subclasses must implement kernel_gram method.")
    

    def __call__(
            self, 
            X: Tensor, 
            Y: Tensor, 
        )->Tensor:
        """
        Computes the kernel evaluation k(X, Y) of two d-dimensional tensors
        (with batch support).

        Args:
            X (Tensor): Tensor with shape (... , d).
            Y (Tensor): Tensor with shape (... , d), with (...) same as X.
        
        Returns:
            Tensor: Tensor with shape (...).
        """
        if X.ndim==1 and Y.ndim==1:
            X = X.unsqueeze(0)
            Y = Y.unsqueeze(0)
        return self.gram(X, Y, diag=True)



class LinearKernel(StaticKernel):
    def __init__(
            self, 
            scale:float = 1.0,
        ):
        """
        The euclidean inner product kernel k(x, y) = scale * <x, y> on R^d.

        Args:
            scale (float, optional): Scaling parameter. Defaults to 1.0.
        """
        super().__init__()
        self.scale = scale
    

    def gram(
            self, 
            X: Tensor, 
            Y: Tensor, 
            diag: bool = False, 
        )->Tensor:
        if diag:
            out = torch.einsum('i...k,i...k -> i...', X, Y)
        else:
            out = torch.einsum('i...k,j...k -> ij...', X, Y)
        return self.scale * out
        


class RBFKernel(StaticKernel):
    def __init__(
            self,
            sigma:float = 1.0,
            scale:float = 1.0
        ):
        """
        The RBF kernel k(x, y) = scale *e^(-sigma * |x-y|^2) on R^d.

        Args:
            sigma (float, optional): RBF parameter. Defaults to 1.0.
            scale (float, optional): Scaling parameter. Defaults to 1.0.
        """
        super().__init__()
        self.sigma = sigma
        self.scale = scale
        self.lin_ker = LinearKernel(scale=1.0)
    

    def gram(
            self, 
            X: Tensor, 
            Y: Tensor, 
            diag: bool = False, 
        )-> Tensor:
        if diag:
            diff = X-Y
            norms_squared = self.lin_ker.gram(diff, diff, diag=True) #shape (N1, ...)
        else:
            xx = self.lin_ker.gram(X, X, diag=True) #shape (N1, ...)
            xy = self.lin_ker.gram(X, Y, diag=False) #shape (N1, N2, ...)
            yy = self.lin_ker.gram(Y, Y, diag=True) #shape (N2, ...)
            norms_squared = -2*xy + xx[:, None] + yy[None, :] 

        return self.scale * np.exp(-self.sigma * norms_squared)



class PolyKernel(StaticKernel):
    def __init__(
            self,
            p:int = 2,
            c:float = 1.0,
            scale:float = 1.0
        ):
        """
        The polynomial kernel k(x, y) =  (scale*<x,y> + c)^p on R^d.

        Args:
            p (int, optional): Polynomial degree. Defaults to 2.
            c (float, optional): Polynomial additive constant. Defaults to 1.0.
            scale (float, optional): Scaling parameter for the dot product.
                Defaults to 1.0.
        """
        super().__init__()
        self.p = p
        self.c = c
        self.lin_ker = LinearKernel(scale)
    

    def gram(
            self, 
            X: Tensor, 
            Y: Tensor, 
            diag: bool = False, 
        )->Tensor:
        return (self.lin_ker.gram(X, Y, diag) + self.c)**self.p