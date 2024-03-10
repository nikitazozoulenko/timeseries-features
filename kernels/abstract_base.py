from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import numpy as np
import torch
from joblib import Parallel, delayed
from torch import Tensor


def is_documented_by(original):
    def wrapper(target):
        target.__doc__ = original.__doc__
        return target
    return wrapper


##########################################  |
#### Static kernel k : R^d x R^d -> R ####  |
########################################## \|/


class StaticKernel():
    """Static kernel k : R^d x R^d -> R."""

    def __call__(
            self, 
            X: Tensor, 
            Y: Tensor,
            diag: bool = False,
        )->Tensor:
        """
        Computes the Gram matrix k(X_i, Y_j), or the diagonal k(X_i, Y_i) 
        if diag=True, with batch support in the middle dimension. If X and Y
        are of ndim=1, they are reshaped to (1, d) and (1, d) respectively.

        Args:
            X (Tensor): Tensor of shape (... , d) or (N1, ... , d).
            Y (Tensor): Tensor of shape (... , d) or (N2, ... , d), 
                with (...) same as X.
            diag (bool): If True, only computes the kernel for
                the diagonal pairs k(X_i, Y_i). Defaults to False.
        
        Returns:
            Tensor: Tensor of shape (N1, N2, ...) or (N1, ...) if diag=True.
        """
        if X.ndim==1:
            X = X.unsqueeze(0)
        if Y.ndim==1:
            Y = Y.unsqueeze(0)
        return self._gram(X, Y, diag)
    

    def _gram(
            self, 
            X: Tensor, 
            Y: Tensor, 
            diag: bool = False, 
        )->Tensor:
        """
        Method to be implemented by subclasses. Computes the Gram matrix 
        k(X_i, Y_j), or the diagonal k(X_i, Y_i) if diag=True.

        Args:
            X (Tensor): Tensor with shape (N1, ... , d).
            Y (Tensor): Tensor with shape (N2, ... , d).
            diag (bool, optional): If True, only computes the kernel for the 
                pairs k(X_i, Y_i). Defaults to False.

        Returns:
            Tensor: Tensor with shape (N1, N2, ...) or (N1, ...) if diag=True.
        """
        raise NotImplementedError("Subclasses must implement '_gram' method.")
    

    def time_gram(
            self, 
            X: Tensor, 
            Y: Tensor, 
            diag: bool = False, 
        )->Tensor:
        """
        Outputs k(X^i_s, Y^j_t), with optional diagonal support across
        the batch dimension.

        Args:
            X (Tensor): Tensor with shape (N1, T1, d).
            Y (Tensor): Tensor with shape (N2, T2, d).
            diag (bool, optional): If True, only computes the kernel for the 
                pairs k(X_i, Y_i). Defaults to False.

        Returns:
            Tensor: Tensor with shape (N1, N2, T1, T2) or (N1, T1, T2) if diag=True.
        """
        if diag:
            X = X.permute(1, 0, 2)
            Y = Y.permute(1, 0, 2)
            trans_gram = self(X, Y) # shape (T1, T2, N)
            return trans_gram.permute(2, 0, 1)
        else:
            N1, T1, d = X.shape
            N2, T2, d = Y.shape
            X = X.reshape(-1, d)
            Y = Y.reshape(-1, d)
            flat_gram = self(X, Y) # shape (N1 * T1, N2 * T2)
            gram = flat_gram.reshape(N1, T1, N2, T2)
            return gram.permute(0, 2, 1, 3)
    

##################################################################  |
#### Time series kernels k : R^(T1 x d) x R^(T2 x d) -> (...) ####  |
################################################################## \|/
        

class TimeSeriesKernel():
    """Time series kernel k : R^(T x d) x R^(T x d) -> (...)"""
    def __init__(
            self,
            max_batch: int = 1000,
            normalize: bool = False,
        ):
        self.max_batch = max_batch
        self.normalize = normalize


    @property
    def log_space(self):
        return False


    def _batched_ker(
            self, 
            X: Tensor, 
            Y: Tensor, 
        ):
        """
        Method to be implemented by subclass. Computes the batched Gram matrix 
        k(X_i, Y_i) for time series X_i and Y_i.

        Args:
            X (Tensor): Tensor with shape (N, T, d).
            Y (Tensor): Tensor with shape (N, T, d).

        Returns:
            Tensor: Tensor with shape (N, ...) where (...) is the dimension 
                of the kernel output.
        """
        raise NotImplementedError("Subclasses must implement '_batched_ker' method.")


    def _max_batched_gram(
            self,
            X: Tensor,
            Y: Tensor,
            diag: bool,
            max_batch: Optional[int],
            normalize: Optional[bool],
            n_jobs: int,
        ):
        """
        Computes the Gram matrix k(X_i, Y_j) for time series X_i and Y_j, 
        or the diagonal k(X_i, Y_i) if diag=True.

        Args:
            X (Tensor): Tensor with shape (N1, T, d) or (T,d).
            Y (Tensor): Tensor with shape (N2, T, d) or (T,d).
            diag (bool): If True, only computes the kernel for the pairs
                k(X_i, Y_i). Defaults to False.
            max_batch (Optional[int]): Sets the max batch size if not None, 
                else uses the default 'self.max_batch'.
            normalize (Optional[int]): If True and diag=False, the kernel is normalized 
                to have unit diagonal via  K(X, Y) = K(X, Y) / sqrt(K(X, X) * K(Y, Y)), 
                and if None defaults to 'self.normalize'.
            n_jobs (int): Number of parallel jobs to run in joblib.Parallel.
        
        Returns:
            Tensor: Tensor with shape (N1, N2, ...) or (N1, ...) if diag=True,
                where (...) is the dimension of the kernel output.
        """
        N1, T, d = X.shape
        N2, _, _ = Y.shape
        device = X.device
        max_batch = max_batch if max_batch is not None else self.max_batch
        normalize = normalize if normalize is not None else self.normalize

        # get indices pairs
        if diag:
            indices = torch.arange(N1, device=device).tile(2,1) # shape (2, N)
        elif X is Y:
            indices = torch.triu_indices(N1, N1, device=device) #shape (2, N*(N+1)//2)
        else:
            indices = torch.cartesian_prod(torch.arange(N1, device=device), 
                                           torch.arange(N2, device=device)).T #shape (2, N1*N2)

        # split into batches
        split = torch.split(indices, max_batch, dim=1)
        result = Parallel(n_jobs=n_jobs)(
            delayed(self._batched_ker)(X[ix], Y[iy])  #self._gram(X[ix], Y[iy], diag=True)
            for ix,iy in split)
        result = torch.cat(result, dim=0)
        extra = result[0].shape

        # reshape back
        if diag:
            result = result.reshape( (N1,) + extra )
        elif X is Y:
            populate = torch.empty((N1, N1) + extra, device=device, dtype=X.dtype)
            for i, (ix, iy) in enumerate(indices.T):
                populate[ix, iy] = result[i]
                populate[iy, ix] = result[i]
            result = populate
        else:
            result = result.reshape( (N1, N2) + extra )
    
        # normalize
        if normalize:
            if diag:
                XX = self._max_batched_gram(X, X, True, max_batch, False, n_jobs) #shape (N, ...)
                YY = self._max_batched_gram(Y, Y, True, max_batch, False, n_jobs) #shape (N, ...)
                if self.log_space:
                    result = result - 0.5*(XX + YY) #shape (N, ...)
                else:
                    result = result / torch.sqrt(XX) / torch.sqrt(YY)
            
            elif X is Y:
                diagonal = torch.einsum('ii...->i...', result) #shape (N, ...)
                if self.log_space:
                    result = result - 0.5*(diagonal[:, None] + diagonal[None, :]) #shape (N, N, ...)
                else:
                    result = result / torch.sqrt(diagonal[:, None]) / torch.sqrt(diagonal[None, :])
            
            else:
                XX = self._max_batched_gram(X, X, True, max_batch, False, n_jobs) #shape (N1, ...)
                YY = self._max_batched_gram(Y, Y, True, max_batch, False, n_jobs) #shape (N2, ...)
                XX = XX[:, None]
                YY = YY[None, :]
                if self.log_space:
                    result = result - 0.5*(XX + YY) #shape (N1, N2, ...) or (N1, ...) if diag==True
                else:
                    result = result / torch.sqrt(XX) / torch.sqrt(YY)
                    
        return result


    @is_documented_by(_max_batched_gram)
    def __call__(
            self, 
            X: Tensor, 
            Y: Tensor,
            diag: bool = False,
            max_batch: Optional[int] = None,
            normalize: Optional[bool] = None,
            n_jobs: int = 1,
        )->Tensor:

        # Reshape
        if X.ndim==2:
            X = X.unsqueeze(0)
        if Y.ndim==2:
            Y = Y.unsqueeze(0)

        # Compute and exponentiate if in log space
        result = self._max_batched_gram(X, Y, diag, max_batch, normalize, n_jobs)
        if self.log_space:
            result = torch.exp(result)

        return result