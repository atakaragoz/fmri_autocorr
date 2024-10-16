import jax
import jax.numpy as jnp
import numpy as np
import os
import nibabel as nib
from functools import partial


def create_acf_jax(lag, n):
    """
    Create a function to compute the autocorrelation function (ACF) using JAX.

    This function returns a JAX-compatible function that calculates the
    autocorrelation for a given lag and series length. The returned function
    is JIT-compiled for improved performance.

    Args:
        lag (int): The lag at which to compute the autocorrelation.
        n (int): The length of the time series.

    Returns:
        function: A JIT-compiled JAX function that computes the autocorrelation
                  for the specified lag.

    Note:
        The returned function expects a 1D JAX array as input and computes
        the autocorrelation using the formula:
        ACF(lag) = Cov(x[t], x[t+lag]) / Var(x)

        Where Cov is the covariance and Var is the variance of the entire series.
    """

    @jax.jit
    def acf_jax(x):
        # Use constants for slicing
        y1 = x[: (n - lag)] - jnp.mean(x)
        y2 = x[lag:] - jnp.mean(x)
        return jnp.sum(y1 * y2) / (n * jnp.var(x))

    return acf_jax


@partial(jax.jit, static_argnums=(1,))
def create_windows(img_clean_jax, window_length):
    """
    Generate sliding windows from the input image data.

    This function takes a 2D array of voxel time series data and creates
    sliding windows of a specified length. The windows are created for
    each voxel and each possible starting time point.

    Args:
        img_clean_jax (jax.numpy.ndarray): A 2D array of shape (n_voxels, T)
            where n_voxels is the number of voxels and T is the number of
            time points.
        window_length (int): The length of each window.

    Returns:
        jax.numpy.ndarray: A 3D array of shape (n_voxels, n_windows, window_length)
            containing the windowed data, where n_windows is T - window_length + 1.

    Note:
        This function is designed to be JIT-compiled with JAX, with window_length
        as a static argument.
    """
    n_voxels, T = img_clean_jax.shape
    n_windows = T - window_length + 1

    # Create an index array for the start of each window
    window_starts = jnp.arange(n_windows)[:, None]

    # Create an index array for the offsets within each window
    window_offsets = jnp.arange(window_length)

    # Combine these to get indices for all windows
    indices = window_starts + window_offsets

    # Use advanced indexing to create the windows
    windows = img_clean_jax[:, None, indices]
    windows = windows.squeeze()
    return windows


@jax.jit
def get_acw_sing_vox(sing_acf_result, TR, acf_length):
    """
    Compute the Autocorrelation Window (ACW) for a single voxel.

    This function calculates the Autocorrelation Window (ACW) for a given voxel's
    autocorrelation function (ACF) result. The ACW is the time point at which
    the ACF first becomes negative, indicating the transition from positive to
    negative autocorrelation. If the ACF does not become negative, the ACW is
    set to the length of the ACF.

    Args:
        sing_acf_result (jax.numpy.ndarray): A 1D array containing the ACF values
            for a single voxel.
        TR (float): The repetition time of the fMRI acquisition, used to convert
            the ACW from indices to seconds.
        acf_length (int): The length of the ACF, used to determine the maximum
            possible ACW value.

    Returns:
        float: The Autocorrelation Window (ACW) in seconds for the given voxel.
    """
    # Create a mask where True indicates sing_acf_result < 0
    mask = sing_acf_result < 0

    # Find the index of the first True value, or acf_length if none exist
    first_negative_index = jnp.argmax(mask)

    # Use jnp.where to select between first_negative_index and acf_length
    item_index = jnp.where(jnp.any(mask), first_negative_index, acf_length)

    # Compute acw
    acw = TR * (item_index - 1)

    return acw


@partial(jax.jit, static_argnums=(1,))
def get_ints_sing_vox(sing_acf_result, max_lags, TR):
    """
    Compute the Integral (INTS) for a single voxel.

    This function calculates the Integral (INTS) for a given voxel's
    autocorrelation function (ACF) result. The INTS is the cumulative sum of
    the ACF values up to the first negative value, indicating the transition
    from positive to negative autocorrelation. If the ACF does not become
    negative, the INTS is set to the cumulative sum of the entire ACF.

    Args:
        sing_acf_result (jax.numpy.ndarray): A 1D array containing the ACF values
            for a single voxel.
        max_lags (int): The maximum number of lags to consider for the INTS
            calculation.
        TR (float): The repetition time of the fMRI acquisition, used to convert
            the INTS from indices to seconds.

    Returns:
        float: The Integral (INTS) in seconds for the given voxel.
    """
    # Create a mask where True indicates sing_acf_result < 0
    mask = sing_acf_result < 0

    # Find the index of the first True value, or acf_length if none exist
    first_negative_index = jnp.argmax(mask)

    # Use jnp.where to select between first_negative_index and acf_length
    item_index = jnp.where(jnp.any(mask), first_negative_index, max_lags)
    # Create a mask of 1s up to item_index (excluding index 0)
    mask2 = jnp.arange(1, max_lags) < item_index

    # Pad mask2 with a leading 0 to match the length of sing_acf_result
    mask2 = jnp.pad(mask2, (1, 0), constant_values=0)

    # Multiply sing_acf_result by the mask and compute the cumulative sum
    masked_result = sing_acf_result * mask2
    ints = TR * jnp.sum(masked_result[1:])  # Exclude the first element (lag 0)

    return ints
