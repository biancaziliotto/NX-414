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
        Per-unit noise ceilings in percent with shape (n_units) or
        (n_channels, n_timepoints), depending on your implementation.
    """

    ### TODO
    # 1:
    z_mu = np.mean(responses,axis=2,keepdims=True)
    z_sigma = np.std(responses,axis=2,keepdims=True)
    z_responses = (responses-z_mu)/z_sigma

    # 2:
    noise_var_per_stim = np.var(z_responses,axis=3)
    noise_var = np.average(noise_var_per_stim, axis = 2)

    # 3:
    signal_var = 1 - noise_var

    # 4:
    snr = signal_var/noise_var
    nc1 = 100 * (signal_var/(signal_var + noise_var))

    
    nc2 = 100 * (snr**2 /(snr**2 + 1/80))
    return nc2


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
        Base RNG seed; each fold may use seed + fold_idx.
    spearman_brown : bool, default=True
        Apply Spearman-Brown correction:
            r_sb = 2r / (1 + r)
    equalize_halves : bool, default=True
        If True, use equal-sized halves and drop one trial if n_reps is odd.
        If False, the second half may be larger by one trial.
    clip_folds : bool, default=False
        If True, clip reliability values after correction.

    Returns
    -------
    np.ndarray
        Array of shape (n_units) or (n_channels, n_timepoints).
    """
    ### TODO
    
    rep_axis = -1
    stim_axis = -2

    n_reps = responses.shape[rep_axis]
    out_shape = responses.shape[:-2]

    fold_rs = np.zeros((folds, *out_shape))

    for f in range(folds):
        rng = np.random.default_rng(seed + f)

        # --- handle odd number of repetitions ---
        if equalize_halves and (n_reps % 2 != 0):
            valid_reps = n_reps - 1
        else:
            valid_reps = n_reps

        all_idx = np.arange(n_reps)

        # --- sample first half ---
        half_size = valid_reps // 2
        idx1 = np.sort(rng.choice(all_idx, size=half_size, replace=False))

        # --- second half = remaining indices ---
        idx2 = np.setdiff1d(all_idx, idx1, assume_unique=True)

        # if we dropped one trial, trim idx2
        if equalize_halves and (n_reps % 2 != 0):
            idx2 = idx2[:half_size]

        # --- split data ---
        slicer = [slice(None)] * responses.ndim

        slicer[rep_axis] = idx1
        half1 = responses[tuple(slicer)]

        slicer[rep_axis] = idx2
        half2 = responses[tuple(slicer)]

        # --- average over repetitions ---
        half1_avg = np.mean(half1, axis=rep_axis)
        half2_avg = np.mean(half2, axis=rep_axis)
        
        # now stimuli is the LAST axis
        stim_axis = -1
        
        x = half1_avg
        y = half2_avg
        
        x_mean = np.mean(x, axis=stim_axis, keepdims=True)
        y_mean = np.mean(y, axis=stim_axis, keepdims=True)
        
        xm = x - x_mean
        ym = y - y_mean
        
        numerator = np.sum(xm * ym, axis=stim_axis)
        denominator = (
            np.sqrt(np.sum(xm**2, axis=stim_axis)) *
            np.sqrt(np.sum(ym**2, axis=stim_axis))
        )
        
        r = numerator / denominator

        # --- Spearman-Brown correction ---
        if spearman_brown:
            r = (2 * r) / (1 + r)

        if clip_folds:
            r = np.clip(r, 0, 1)

        fold_rs[f] = r

    return np.mean(fold_rs, axis=0)*100

def random_split(data, axis, eq):
    n = data.shape[axis]
    if eq:
        if n%2 != 0:
            np.delete(data, np.random.randint(n),axis=axis)
            n = n-1
        
    choice = np.random.choice(range(data.shape[axis]), size=(n//2,), replace=False)    
    ind = np.zeros(data.shape[axis], dtype=bool)
    ind[choice] = True
    rest = ~ind
    return data[:,:,:,ind], data[:,:,:,rest]

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