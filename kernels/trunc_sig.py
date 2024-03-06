


############################################################################
################# signature kernels of static kernels ######################
############################################################################


def sig_kernel(s1:Tensor, 
               s2:Tensor, 
               order:int,
               static_kernel_gram:Callable = linear_kernel_gram,
               only_last:bool = True):
    """Computes the truncated signature kernel of two time series of 
    shape (T_i, d) with respect to a static kernel on R^d.

    Args:
        s1 (np.ndarray): Array of shape (T_1, d).
        s2 (np.ndarray): Array of shape (T_2, d).
        order (int): Truncation order of the signature kernel.
        static_kernel_gram (Callable): Gram kernel function taking in two ndarrays,
                            see e.g. 'linear_kernel_gram' or 'rbf_kernel_gram'.
        only_last (bool): If False, returns results of all truncation levels up to 'order'.
    """
    K = static_kernel_gram(s1, s2)
    nabla = K[1:, 1:] + K[:-1, :-1] - K[1:, :-1] - K[:-1, 1:]
    sig_kers = jitted_trunc_sig_kernel(nabla, order)
    if only_last:
        return sig_kers[-1]
    else:
        return sig_kers



@njit((nb.float64[:, ::1], nb.int64), fastmath=True, cache=True)
def reverse_cumsum(arr:Tensor, axis:int): #ndim=2
    """JITed reverse cumulative sum along the specified axis.
    (np.cumsum with axis is not natively supported by Numba)
    
    Args:
        arr (np.ndarray): Array of shape (T_1, T_2).
        axis (int): Axis along which to cumsum.
    """
    A = arr.copy()
    if axis==0:
        for i in np.arange(A.shape[0]-2, -1, -1):
            A[i, :] += A[i+1, :]
    else: #axis==1
        for i in np.arange(A.shape[1]-2, -1, -1):
            A[:,i] += A[:,i+1]
    return A



@njit((nb.float64[:, ::1], nb.int64), fastmath=True, cache=True)
def jitted_trunc_sig_kernel(nabla, order):
    """Given difference matrix nabla_ij = K[i+1, j+1] + K[i, j] - K[i+1, j] - K[i, j+1],
    computes the truncated signature kernel of all orders up to 'order'."""
    B = np.ones((order+1, order+1, order+1, *nabla.shape))
    for d in np.arange(order):
        for n in np.arange(order-d):
            for m in np.arange(order-d):
                B[d+1,n,m] = 1 + nabla/(n+1)/(m+1)*B[d, n+1, m+1]
                r1 = reverse_cumsum(nabla * B[d, n+1, 1] / (n+1), axis=0)
                B[d+1,n,m, :-1, :] += r1[1:, :]
                r2 = reverse_cumsum(nabla * B[d, 1, m+1] / (m+1), axis=1)
                B[d+1,n,m, :, :-1] += r2[:, 1:]
                rr = reverse_cumsum(nabla * B[d, 1, 1], axis=0)
                rr = reverse_cumsum(rr, axis=1)
                B[d+1,n,m, :-1, :-1] += rr[1:, 1:]

    #copy, otherwise all memory accumulates in for loop
    return B[1:,0,0,0,0].copy() 



def sig_kernel_gram(
        X:List[np.ndarray],
        Y:List[np.ndarray],
        order:int,
        static_kernel_gram:Callable,
        only_last:bool = True,
        sym:bool = False,
        n_jobs:int = 1,
        verbose:bool = False,
    ):
    """Computes the Gram matrix k_sig(X_i, Y_j) of the signature kernel,
    given the static kernel k(x, y) and the truncation order.

    Args:
        X (List[np.ndarray]): List of time series of shape (T_i, d).
        Y (List[np.ndarray]): List of time series of shape (T_j, d).
        order (int): Truncation level of the signature kernel.
        static_kernel_gram (Callable): Gram kernel function taking in two ndarrays,
                            see e.g. 'linear_kernel_gram' or 'rbf_kernel_gram'.
        only_last (bool): If False, returns results of all truncation levels up to 'order'.
        sym (bool): If True, computes the symmetric Gram matrix.
        n_jobs (int): Number of parallel jobs to run.
        verbose (bool): Whether to enable the tqdm progress bar.
    """
    pairwise_ker = lambda s1, s2 : sig_kernel(s1, s2, order, static_kernel_gram, only_last)
    return pairwise_kernel_gram(X, Y, pairwise_ker, sym, n_jobs, verbose)

