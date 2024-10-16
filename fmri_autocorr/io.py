from nilearn.maskers import NiftiMasker
import nibabel as nib
import jax.numpy as jnp
import os


def load_data_and_mask(src_fmri, src_mask):
    """
    Loads fMRI data and its corresponding mask for a given subject and hemisphere.

    Parameters:
    - src_fmri (str): The file path where the fMRI data is located.
    - src_mask (str): The file path where the mask data is located.

    Returns:
    - img_clean (jax.numpy.array): The cleaned and filtered fMRI data.
    - masker (NiftiMasker): The NiftiMasker object used for cleaning and filtering.
    - TR (float): The repetition time (TR) of the fMRI data.
    """

    nii_img = nib.load(src_fmri)
    nii_img_msk = nib.load(src_mask)
    TR = nii_img.header["pixdim"][4]
    # Define filter parameters
    lowcut = 0.01
    highcut = 0.1
    # filter and detrend signal using nilearn.img.clean_img
    masker = NiftiMasker(
        mask_img=nii_img_msk,
        detrend=True,
        low_pass=highcut,
        high_pass=lowcut,
        t_r=TR,
    )
    img_clean = masker.fit_transform(nii_img).T
    return jnp.array(img_clean), masker, TR


def save_out_data(acws, ints, ac1s, masker, sub, hemi, dst_dir):
    """
    Saves the autocorrelation weights (acws), intercepts (ints), and autocorrelation coefficients (ac1s) as Nifti files.

    Parameters:
    - acws (jax.numpy.array): The autocorrelation weights.
    - ints (jax.numpy.array): The intercepts.
    - ac1s (jax.numpy.array): The autocorrelation coefficients.
    - masker (NiftiMasker): The NiftiMasker object used for cleaning and filtering.
    - sub (str): The subject ID.
    - hemi (str): The hemisphere ('L' or 'R') to save data for.
    - dst_dir (str): The destination directory where the data will be saved.
    """
    dst_acw = os.path.join(dst_dir, f"{sub}_{hemi}_acw.nii.gz")
    dst_ints = os.path.join(dst_dir, f"{sub}_{hemi}_ints.nii.gz")
    dst_ac1s = os.path.join(dst_dir, f"{sub}_{hemi}_ac1s.nii.gz")
    masker.inverse_transform(acws).to_filename(dst_acw)
    masker.inverse_transform(ints).to_filename(dst_ints)
    masker.inverse_transform(ac1s.T).to_filename(dst_ac1s)
