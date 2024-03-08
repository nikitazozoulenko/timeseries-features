from typing import List, Dict, Set, Any, Optional, Tuple, Literal, Callable
import numpy as np
import torch
from torch import Tensor
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kernels.abstract_base import StaticKernel, TimeSeriesKernel
from kernels.static_kernels import PolyKernel

#######################################################################################
################### Time series Integral Kernel of static kernel ######################
#######################################################################################

class IntegralKernel(TimeSeriesKernel):
    def __init__(
            self,
            static_kernel:StaticKernel = PolyKernel,
        ):
        """
        The integral kernel K(x, y) = \int k(x_t, y_t) dt, given a static kernel 
        k(x, y) on R^d.

        Args:
            static_kernel (StaticKernel): Static kernel on R^d.
        """
        super().__init__()
        self.static_kernel = static_kernel


    def _gram(
            self, 
            X: Tensor, 
            Y: Tensor, 
            diag: bool = False, 
        ):
        #Shape (N1, N2, T), or (N1, T) if diag=True
        ijKt = self.static_kernel.gram(X, Y, diag)

        #return integral of k(x_t, y_t) dt for each pair x and y
        T = X.shape[-2]
        return torch.trapz(ijKt, dx=1/(T-1), axis=-1)