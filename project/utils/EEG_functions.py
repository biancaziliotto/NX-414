import h5py
import numpy as np
from scipy import stats

def load_subject(data,subject_id,roi):
    '''
    subjects: ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    ROIs: ['central', 'frontal', 'occipital', 'occipital_parietal', 'parietal', 'temporal', 'whole_brain']
    '''
    eeg_noise_ceilings = data['noise_ceilings']
    eeg_noise_ceilings_train = data['noise_ceilings_train']
    eeg_test = data['test']
    eeg_train = data['train']

    return eeg_train['neural_data'][subject_id][roi], eeg_train['stimulus_ids'], eeg_noise_ceilings_train[subject_id][roi]

def _reshape_to_3d(responses):
    """Reshape (n_channels, n_timepoints, n_stimuli, n_reps) to (n_units, n_stimuli, n_reps)."""
    if responses.ndim == 3:
        return responses, None
    elif responses.ndim == 4:
        n_c, n_t, n_s, n_r = responses.shape
        return responses.reshape(n_c * n_t, n_s, n_r), (n_c, n_t)
    else:
        raise ValueError("responses must be 3D or 4D")


def _restore_shape(arr, reshape_info):
    if reshape_info is None:
        return arr
    return arr.reshape(reshape_info)


def compute_ceiling_variancebased(responses: np.ndarray, nan_policy: str = 'omit') -> np.ndarray:
    """
    Noise ceiling per unit using the method described in the NSD paper
    (Allen et al., 2021 / 2022 style variance-based estimator).

    Steps:
      1) z-score across stimuli (axis=1) for each (unit, rep) -> total var ≈ 1
      2) estimate noise variance across repetitions (axis=2), then average across stimuli
      3) signal variance = 1 - noise_var
      4) reliability (percent) for finite repeats:
             nc = 100 * (snr / (snr + 1 / n_reps))

    Parameters
    ----------
    responses : np.ndarray
        Shape (n_units, n_stimuli, n_reps) or
        (n_channels, n_timepoints, n_stimuli, n_reps).
    nan_policy : {'propagate', 'raise', 'omit'}, default='omit'
        Passed to the z-scoring logic when handling NaNs.

    Returns
    -------
    np.ndarray
        Per-unit noise ceilings in percent with shape (n_units,) or
        (n_channels, n_timepoints).
    """
    x, reshape_info = _reshape_to_3d(responses)
    n_units, n_stimuli, n_reps = x.shape

    if n_reps < 2:
        return np.full((n_units,), np.nan)

    if nan_policy == 'omit':
        mean = np.nanmean(x, axis=1, keepdims=True)
        std = np.nanstd(x, axis=1, ddof=1, keepdims=True)
    elif nan_policy == 'propagate':
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, ddof=1, keepdims=True)
    elif nan_policy == 'raise':
        if np.isnan(x).any():
            raise ValueError("NaNs present in input")
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, ddof=1, keepdims=True)
    else:
        raise ValueError(f"Invalid nan_policy: {nan_policy!r}")

    std = np.where(std == 0, np.nan, std)
    z = (x - mean) / std

    if nan_policy == 'omit':
        noise_var = np.nanmean(np.nanvar(z, axis=2, ddof=1), axis=1)
    else:
        noise_var = np.mean(np.var(z, axis=2, ddof=1), axis=1)

    signal_var = np.maximum(1.0 - noise_var, 0)
    snr = signal_var / noise_var
    nc = 100.0 * (snr / (snr + 1.0 / n_reps))

    return _restore_shape(nc, reshape_info)


def compute_ceiling_splithalf(
    responses: np.ndarray,
    folds: int = 10,
    seed: int = 0,
    spearman_brown: bool = True,
    equalize_halves: bool = True,
    clip_folds: bool = False
) -> np.ndarray:
    """
    Split-half reliability per unit (voxel / channel / channel*timepoint).
    You can refer to van Bree et al. (2025) for mathematical details.

    Steps:
      1) For each fold, randomly split repetitions into two halves.
      2) Average responses within each half and compute Pearson correlation across stimuli.
      3) Optionally apply Spearman-Brown correction to each fold's correlation.
      4) Average across folds to get a final reliability estimate.

    Parameters
    ----------
    responses : np.ndarray
        Shape (n_units, n_stimuli, n_reps) or
        (n_channels, n_timepoints, n_stimuli, n_reps).
        The last axis corresponds to repetitions / trials.
    folds : int, default=10
        Number of random split-halves to sample.
    seed : int, default=0
        RNG seed.
    spearman_brown : bool, default=True
        Apply Spearman-Brown correction: r_sb = 2r / (1 + r).
    equalize_halves : bool, default=True
        If True, use equal-sized halves (drops one trial if n_reps is odd).
    clip_folds : bool, default=False
        If True, clip reliability values to [-1, 1] after correction.

    Returns
    -------
    np.ndarray
        Array of shape (n_units,) or (n_channels, n_timepoints).
    """
    x, reshape_info = _reshape_to_3d(responses)
    n_units, n_stimuli, n_reps = x.shape

    if n_reps < 2:
        return np.full((n_units,), np.nan)

    rng = np.random.RandomState(seed)
    all_fold_r = []

    for _ in range(folds):
        perm = rng.permutation(n_reps)
        half = n_reps // 2
        idx1 = perm[:half]
        idx2 = perm[half:2 * half] if equalize_halves else perm[half:]

        x1 = np.nanmean(x[:, :, idx1], axis=2)
        x2 = np.nanmean(x[:, :, idx2], axis=2)

        x1c = x1 - np.nanmean(x1, axis=1, keepdims=True)
        x2c = x2 - np.nanmean(x2, axis=1, keepdims=True)

        num = np.nansum(x1c * x2c, axis=1)
        den = np.sqrt(np.nansum(x1c ** 2, axis=1) * np.nansum(x2c ** 2, axis=1))
        r = num / den

        if spearman_brown:
            r = 2 * r / (1 + r)
        if clip_folds:
            r = np.clip(r, -1, 1)

        all_fold_r.append(r)

    rel = np.nanmean(np.stack(all_fold_r, axis=0), axis=0)
    return _restore_shape(rel * 100, reshape_info)


def compare_noise_ceilings(
    h5_path: str,
    compute_fn
):
    """
    Compare stored noise ceilings with a custom estimator using MSE.

    Parameters
    ----------
    h5_path : str
        Path to HDF5 dataset.
    compute_fn : callable
        Your function, e.g. compute_ceiling_variancebased or split-half version.
        Must return (channels, timepoints).


    Returns
    -------
    results : dict
        Nested dict: results[subject][region] = mse
    """

    results = {}

    with h5py.File(h5_path, "r") as f:
        subjects = list(f["noise_ceilings"].keys())
        print(subjects)

        for subj in subjects:
            results[subj] = {}
            
            regions = list(f["noise_ceilings"][subj].keys())
            print(regions)
            for region in regions:
                # --- Load reference ceilings ---
                ref = np.asarray(f["noise_ceilings"][subj][region])  # (channels, time)

                # --- Load neural data ---
                data = np.asarray(h5py.File('/shared/NX-414/data/things_eeg2-test_reps.h5', 'r')["test"]["neural_data"][subj][region])
                print(data.shape)
                # --- Ensure correct shape ---
                # Expected: (channels, time, stimuli, reps)
                if data.shape != (*ref.shape, data.shape[-2], data.shape[-1]):
                    # Try common case: (stimuli, reps, channels, time)
                    if data.ndim == 4:
                        data = np.transpose(data, (1, 2, 0, 3))
                    else:
                        raise ValueError(f"Unexpected shape for {subj}-{region}: {data.shape}")

                # --- Compute your ceiling ---
                est = compute_fn(data)

                # --- Compute MSE ---
                mse = np.nanmean((est - ref) ** 2)

                results[subj][region] = mse

    return results