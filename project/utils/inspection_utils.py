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