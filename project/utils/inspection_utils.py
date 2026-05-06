import os
import h5py

DATA_ROOT = os.environ.get("NX414_DATA_ROOT", "/shared/NX-414")

def inspect(data, key="path", level=0, verbose=False):

    if hasattr(data, "keys"):
        print("  "*level, key, f" ({len(list(data.keys()))}) :", list(data.keys()))
        for k in list(data.keys()):
            inspect(data[k], k, level+1, verbose=verbose)

    elif verbose:
        print("  "*level, f"{key} - {data.shape}")

    return

def load_and_inspect_h5(path, verbose=False):
    with h5py.File(path, "r") as f:
        inspect(f, path, verbose=verbose)
    return

def load_tsvd_dataset(split="train", subject="monkeyF", roi="V1", data_root=None):
    """
    Loads the TVSD dataset for a given subject and ROI.
    Parameters:
    - subject (str): Subject identifier (e.g., "monkeyF").
    - roi (str): Region of interest (e.g., "V1").
    - data_root (str): Root directory containing data/ and extracted_features/.
    Returns:
    - np.ndarray: Neural response data for the specified subject and ROI. shape: (n_stim, n_units)
    """
    root = data_root or DATA_ROOT
    with h5py.File(os.path.join(root, "data", "tvsd.h5"), "r") as f:
        data = f[split]["neural_data"][subject][roi][:]
        stimuli = f[split]["stimulus_ids"][:]

    return data, stimuli

def load_eeg2_dataset(split="train", subject="sub-01", roi="occipital", data_root=None):
    """
    Loads the EEG2 dataset for a given subject and ROI.
    Parameters:
    - subject (str): Subject identifier (e.g., "sub-01").
    - roi (str): Region of interest (e.g., "occipital").
    - data_root (str): Root directory containing data/ and extracted_features/.
    Returns:
    - np.ndarray: Neural response data for the specified subject and ROI, averaging across timepoints. shape: (n_stim, n_channels)
    """
    root = data_root or DATA_ROOT
    with h5py.File(os.path.join(root, "data", "things_eeg2.h5"), "r") as f:
        data = f[split]["neural_data"][subject][roi][:] .mean(axis=-1)  # Average across timepoints
        stimuli = f[split]["stimulus_ids"][:]

    return data, stimuli


def load_nsd_dataset(split="train", subject="subj01", roi="V1d", data_root=None):
    """
    Loads the NSD dataset for a given subject and ROI.
    Parameters:
    - subject (str): Subject identifier (e.g., "subj01").
    - roi (str): Region of interest (e.g., "V1d").
    - data_root (str): Root directory containing data/ and extracted_features/.
    Returns:
    - np.ndarray: Neural response data for the specified subject and ROI. shape: (n_stim, n_voxels)
    """
    root = data_root or DATA_ROOT
    with h5py.File(os.path.join(root, "data", "nsd_func1pt8mm_individualROIs.h5"), "r") as f:
        data = f[split]["neural_data"][subject][roi][:]
        stimuli = f[split]["stimulus_ids"][subject][:]

    return data, stimuli
