import h5py

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

def load_tsvd_dataset(split="train", subject="MonkeyF", roi="V1"):
    """
    Loads the EEG dataset for a given subject and ROI.
    Parameters:
    - subject (str): Subject identifier (e.g., "subj-01").
    - roi (str): Region of interest (e.g., "V1").
    Returns:
    - np.ndarray: Neural response data for the specified subject and ROI. shape: (n_stim, n_units)
    """
    with h5py.File("/shared/NX-414/data/tvsd.h5", "r") as f:
         data = f[split][subject][roi]

    return data

def load_eeg_dataset(split="train", subject="subj-01", roi="occipital"):
    """
    Loads the EEG dataset for a given subject and ROI.
    Parameters:
    - subject (str): Subject identifier (e.g., "subj-01").
    - roi (str): Region of interest (e.g., "V1").
    Returns:
    - np.ndarray: Neural response data for the specified subject and ROI, averaging across timepoints. shape: (n_stim, n_channels)
    """
    with h5py.File("/shared/NX-414/data/things_eeg2.h5", "r") as f:
         data = f[split]["neural_data"][subject][roi].mean(axis=-1)  # Average across timepoints

    return data

def load_nsd_dataset(subject="subj01", roi="V1d"):
    """
    Loads the NSD dataset for a given subject and ROI.
    Parameters:
    - subject (str): Subject identifier (e.g., "subj01").
    - roi (str): Region of interest (e.g., "V1d").
    Returns:
    - np.ndarray: Neural response data for the specified subject and ROI. shape: (n_stim, n_voxels)
    """
    with h5py.File("/shared/NX-414/data/nsd_func1pt8mm_individualROIs.h5", "r") as f:
         data = f[subject]["neural_data"][subject][roi]
    
    return data
