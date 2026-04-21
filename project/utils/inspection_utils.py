import h5py

def inspect(data, key="path"):
    if not hasattr(data, "keys"):
        print(f"{key} - {type(data)}")
    else:
        print("\t")
        print(key)
        for k in list(data.keys()):
            inspect(data[k], k)