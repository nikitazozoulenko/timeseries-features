from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import numpy as np
import torch
from torch import Tensor


##########################################  |
#### Static kernel k : R^d x R^d -> R ####  |
########################################## \|/


class StaticKernel():
    """Static kernel k : R^d x R^d -> R."""
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
        raise NotImplementedError("Subclasses must implement 'gram' method.")
    

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
    

##############################################################  |
#### Time series kernels k : R^(T1 x d) x R^(T2 x d) -> R ####  |
############################################################## \|/
    

class TimeSeriesKernel():
    """Time series kernel k : R^(T x d) x R^(T x d) -> R"""
    def __init__(
            self,
            max_batch: Optional[int] = None,
            normalize: bool = False,
        ):
        self.max_batch = max_batch
        self.normalize = normalize


    def _gram(
            self, 
            X: Tensor, 
            Y: Tensor, 
            diag: bool = False,
        ):
        """
        Computes the Gram matrix k(X_i, Y_j) for time series X_i and Y_j, 
        or the diagonal k(X_i, Y_i) if diag=True. 

        Args:
            X (Tensor): Tensor with shape (N1, T, d).
            Y (Tensor): Tensor with shape (N2, T, d).
            diag (bool, optional): If True, only computes the kernel for the 
                pairs K(X_i, Y_i). Defaults to False.

        Returns:
            Tensor: Tensor with shape (N1, N2) or (N1) if diag=True.
        """
        raise NotImplementedError("Subclasses must implement 'gram' method.")
    

    def gram(
            self, 
            X: Tensor, 
            Y: Tensor, 
            diag: bool = False,
            max_batch: Optional[int] = None,
            normalize: Optional[bool] = None,
        ):
        """
        Computes the Gram matrix k(X_i, Y_j) for time series X_i and Y_j, 
        or the diagonal k(X_i, Y_i) if diag=True.

        Args:
            X (Tensor): Tensor with shape (N1, T, d).
            Y (Tensor): Tensor with shape (N2, T, d).
            diag (bool): If True, only computes the kernel for the 
                pairs K(X_i, Y_i). Defaults to False.
            max_batch (Optional[int]): Sets the max batch size if not None, 
                else uses the default 'self.max_batch'. Defaults to None.
            normalize (Optional[int]): If True, the kernel is normalized to 
                have unit diagonal via  K(X, Y) = K(X, Y) / sqrt(K(X, X) * K(Y, Y)), 
                and if None defaults to 'self.normalize'.

        Returns:
            Tensor: Tensor with shape (N1, N2) or (N1) if diag=True.
        """
        #TODO add batch support
        #TODO add normalize support
        return self._gram(X, Y, diag)


    def __call__(
            self, 
            X: Tensor, 
            Y: Tensor,
            max_batch: Optional[int] = None,
            normalize: Optional[bool] = None,
        )->Tensor:
        """
        Computes the kernel evaluation k(X, Y) of two time series
        (with batch support).

        Args:
            X (Tensor): Tensor with shape (... , T, d).
            Y (Tensor): Tensor with shape (... , T, d), with (...) same as X.
            max_batch (Optional[int]): Sets the max batch size if not None, 
                else uses the default 'self.max_batch'. Defaults to None.
            normalize (Optional[int]): If True, the kernel is normalized to 
                have unit diagonal via  K(X, Y) = K(X, Y) / sqrt(K(X, X) * K(Y, Y)), 
                and if None defaults to 'self.normalize'.
        
        Returns:
            Tensor: Tensor with shape (...).
        """
        # Reshape
        if X.ndim==2:
            X = X.unsqueeze(0)
        if Y.ndim==2:
            Y = Y.unsqueeze(0)
        original_shape = X.shape
        X = X.reshape(-1, original_shape[-2], original_shape[-1])
        Y = Y.reshape(-1, original_shape[-2], original_shape[-1])
        N, T, d = X.shape

        # Compute kernel
        out = self.gram(X, Y, diag=True, max_batch=max_batch, normalize=normalize)
        return out.reshape(original_shape[:-2])
