#TODO install fbm library

def gen_fBM(H:float, n_samples:int, T:int):
    """
    Generate fractional Brownian motion (fBM) with Hurst exponent 'H'.
    
    Args:
        H (float): Hurst exponent.
        n_samples (int): Number of samples.
        T (int): Number of time steps.
    
    Returns: 
        Tensor with shape (N, T).
    """
    #
    return fbm(n=T, hurst=H, length=T, method='daviesharte')