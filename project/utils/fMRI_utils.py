import nibabel as nib
import nilearn
from nilearn import datasets, plotting, surface
from nilearn.surface import load_surf_mesh
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

def load_mhg(mhg_path):
    
    img = nib.load(mhg_path)
    data = np.squeeze(img.get_fdata())
    
    return data

def downsample_to_fs5(data, hemi):

    fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage")
    fsavg5 = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    src_coords, _ = load_surf_mesh(fsavg[f"infl_{hemi}"])
    trg_coords, _ = load_surf_mesh(fsavg5[f"infl_{hemi}"])

    data = np.squeeze(data)

    tree = cKDTree(src_coords)
    _, idx = tree.query(trg_coords)

    data_downsampled = data[idx]

    return data_downsampled, fsavg5

def ncsnr_to_NC(ncsnr, n):
    """
    Using formula from Allen 2021.
    """
    return 100 * ncsnr**2 / (ncsnr**2 + 1/n)

def plot_hist(data, output_file):
    fig, axs = plt.subplots()
    axs.hist(data.flatten(), bins=100)
    fig.savefig(output_file)

    return

def parcel_mean(data, labels):

    parcel_ids = np.unique(labels)
    
    means = []
    ids = []

    for pid in parcel_ids:
        if pid == 0:  # background
            continue
        mask = labels == pid
        if np.any(mask):
            means.append(np.mean(data[mask]))
            ids.append(pid)

    return np.array(ids), np.array(means)

def plot_parcel_summary(data, hemi, output_file):
    
    destrieux = datasets.fetch_atlas_surf_destrieux()
    data_downsampled, fsaverage5 = downsample_to_fs5(data, hemi)
    
    if output_file is None:
        output_file = f"figures/parcel_summary_{hemi}.png"
    
    labels = destrieux['labels']
    
    ids, mean = parcel_mean(data_downsampled, destrieux[f'map_{hemi}'])
    
    def id_to_name(ids, labels):
        return [labels[i] for i in ids]

    names = id_to_name(ids, labels)
    
    fig, axs = plt.subplots(figsize = (10,12))
    
    order = np.argsort(mean)[::-1]
    
    axs.bar(np.array(names)[order][:15], mean[order][:15])
    axs.tick_params(axis='x', labelrotation=90)
    axs.set_title("Top Destrieux parcels")
    axs.set_ylabel("Noise Ceiling (%)")
    
    fig.savefig(output_file)
    
    return

def plot_on_surface(data, hemi="left", brain_area='infl_left', output_file=None, parcel_overlay=False):
    
    destrieux = datasets.fetch_atlas_surf_destrieux()
    if parcel_overlay:
        data, fsaverage = downsample_to_fs5(data, hemi)
    else:
        fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage")
    
    if brain_area not in list(fsaverage.keys()):
        print(f"please specify one of the following brain areas: {list(fsaverage.keys())}")
        raise
    if brain_area.split("_")[-1] != hemi:
        print(f"Ensure brain area and hemisphere are consistent: {hemi} and {brain_area}")
        raise
        
    if output_file is None:
        output_file = f"figures/{brain_area}.png"
    
    fig = plotting.plot_surf_stat_map(
        fsaverage[brain_area],
        data,
        hemi=hemi,
        title=f'{hemi} hemisphere',
        colorbar=True,
    )
    if parcel_overlay:
        plotting.plot_surf_contours(
            fsaverage[brain_area],
            destrieux[f'map_{hemi}'],
            levels=[i for i in set(destrieux[f'map_{hemi}']) if i != 0],
            figure=fig,
        )
    fig.savefig(output_file, dpi=300)