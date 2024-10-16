from fmri_autocorr.fmri_autocorr import *
from fmri_autocorr.io import *
import argparse
import time


def main(args):
    # Move the setup code inside the main function
    window_length = args.window_length - 1
    acf_length = args.acf_length - 1
    max_lags = acf_length
    n = window_length
    roi = args.roi

    # Create acf_funcs inside main to use the correct window_length
    acf_funcs = [create_acf_jax(lag, n) for lag in range(max_lags)]

    # JIT compile the compute_all_lags function
    @jax.jit
    def compute_all_lags(window):
        return jnp.array([func(window) for func in acf_funcs])

    # Create vmapped versions of functions
    compute_all_lags_vmap = jax.vmap(jax.vmap(compute_all_lags, in_axes=0), in_axes=0)
    get_acw_vmap = jax.vmap(get_acw_sing_vox, in_axes=(0, None, None))
    get_acw_vmap2 = jax.vmap(get_acw_vmap, in_axes=(1, None, None))
    get_ints_vmap = jax.vmap(get_ints_sing_vox, in_axes=(0, None, None))
    get_ints_vmap2 = jax.vmap(get_ints_vmap, in_axes=(1, None, None))

    # Read in subject list
    sublist = np.loadtxt(args.subject_list, dtype="str")

    for sub in sublist:

        img_clean_jax, masker, TR = load_data_and_mask(args.src_fmri, args.src_mask)

        windows = create_windows(img_clean_jax, window_length)
        results = compute_all_lags_vmap(windows)

        acws = get_acw_vmap2(results, TR, acf_length)
        ints = get_ints_vmap2(results, max_lags, TR)
        ac1s = results[:, :, 1]  # get the first lag from each window

        save_out_data(acws, ints, ac1s, masker, sub, roi, args.dst_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dAC analysis on fMRI data.")
    parser.add_argument(
        "--window_length", type=int, default=30, help="Window length for analysis"
    )
    parser.add_argument(
        "--acf_length", type=int, default=30, help="ACF length for analysis"
    )
    parser.add_argument(
        "--src_fmri", type=str, required=True, help="Source file path for fMRI data"
    )
    parser.add_argument(
        "--src_mask", type=str, required=True, help="Source file path for mask data"
    )
    parser.add_argument(
        "--dst_dir", type=str, required=True, help="Destination directory for output"
    )
    parser.add_argument(
        "--subject_list", type=str, required=True, help="Path to subject list file"
    )
    parser.add_argument("--roi", type=str, required=True, help="ROI name for output")

    args = parser.parse_args()

    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")
