#!/usr/bin/env python
"""
Augments existing result JSONs in ./results/ with metrics that were not
computed during the original training run:

  Per-unit predictive metrics (summarised as mean/std/median/min/max):
    - Pearson correlation
    - Noise-corrected Pearson correlation
    - Explained variance
    - Noise-corrected explained variance
    - Noise ceiling (mean only)

  Hybrid representational metrics (single value per layer):
    - encoding_rsa  : RSA between predicted and actual neural responses
    - encoding_cka  : CKA between predicted and actual neural responses

  Encoding-RSA/CKA measure whether the *geometry* of predicted responses
  matches the geometry of actual responses, allowing direct comparison with
  the feature-space RSA/CKA scores computed in alignement_utils.

For every layer entry missing any of these keys the script reloads the saved
model weights, re-fits the same StandardScalers used during training
(deterministic), runs inference on the test set, and writes the metrics back
to the JSON.  The script is idempotent: entries that already have all keys
are skipped.

Usage:
    python augment_results.py [--results-dir ./results] [--base-dir .]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from utils.predictive_alignement import LinearRegressionModel
from utils.inspection_utils import load_tsvd_dataset, load_eeg2_dataset, load_nsd_dataset
from utils.evaluation_metrics import (
    compute_pearson_correlation,
    compute_explained_variance,
    compute_noise_ceiling,
    compute_noise_corrected_pearson,
    compute_noise_corrected_explained_variance,
)
from utils.alignement_utils import RepresentationalSimilarityAnalysis, CenteredKernelAlignment

ADDED_KEYS = {
    # per-unit predictive metrics
    "pearson_corr_mean", "pearson_corr_std", "pearson_corr_median",
    "pearson_corr_min", "pearson_corr_max",
    "explained_var_mean", "explained_var_std", "explained_var_median",
    "explained_var_min", "explained_var_max",
    "noise_corrected_pearson_mean", "noise_corrected_pearson_std",
    "noise_corrected_pearson_median", "noise_corrected_pearson_min",
    "noise_corrected_pearson_max",
    "noise_corrected_ev_mean", "noise_corrected_ev_std",
    "noise_corrected_ev_median", "noise_corrected_ev_min",
    "noise_corrected_ev_max",
    "noise_ceiling_mean",
    # hybrid representational metrics
    "encoding_rsa",
    "encoding_cka",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_neural(neural_dataset, roi, subject, split):
    nd = neural_dataset.lower()
    if nd in ("tvsd", "things_tvsd"):
        return load_tsvd_dataset(split=split, subject=subject or "monkeyF", roi=roi)
    elif nd in ("eeg2", "things_eeg2"):
        return load_eeg2_dataset(split=split, subject=subject or "sub-01", roi=roi)
    elif nd == "nsd":
        return load_nsd_dataset(split=split, subject=subject or "subj01", roi=roi)
    else:
        raise ValueError(f"Unknown neural dataset: {neural_dataset!r}")


def _load_features(model_name, dataset_name, stimuli_ids, layer_name):
    import h5py
    path = f"/shared/NX-414/extracted_features/{model_name}/{dataset_name}.h5"
    with h5py.File(path, "r") as f:
        feat_ids = f["ids"][:]
        id_to_idx = {id_: i for i, id_ in enumerate(feat_ids)}
        feat_idx = np.array([id_to_idx[x] for x in stimuli_ids])
        sort_order = np.argsort(feat_idx)
        restore_order = np.argsort(sort_order)
        X = f["features"][layer_name][feat_idx[sort_order]][restore_order]
    return X


def _neural_dataset_from_path(weights_file):
    """Parse neural_dataset from the weights file path.

    Path: encoders/{model}/{dataset}/{neural_dataset}/{roi}[/{subject}]/{layer}.pth
    """
    parts = Path(weights_file).parts
    try:
        idx = list(parts).index("encoders")
        return parts[idx + 3]
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _summarise(arr, prefix):
    return {
        f"{prefix}_mean":   float(np.mean(arr)),
        f"{prefix}_std":    float(np.std(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_min":    float(np.min(arr)),
        f"{prefix}_max":    float(np.max(arr)),
    }


def compute_extra_metrics(y_true, y_pred, rsa, cka):
    """
    Compute all added metrics from ground-truth and predicted responses.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_stimuli, n_units)
        Actual neural responses on the test set.
    y_pred : np.ndarray, shape (n_stimuli, n_units)
        Predicted responses from the linear encoding model.
    rsa : RepresentationalSimilarityAnalysis
    cka : CenteredKernelAlignment

    Returns
    -------
    dict  – flat dict of floats, ready to update a results entry.
    """
    noise_ceiling = compute_noise_ceiling(y_true)
    pearson, _    = compute_pearson_correlation(y_true, y_pred)
    ev            = compute_explained_variance(y_true, y_pred)
    nc_pearson    = compute_noise_corrected_pearson(y_true, y_pred, noise_ceiling)
    nc_ev         = compute_noise_corrected_explained_variance(y_true, y_pred, noise_ceiling)

    # Encoding-RSA and encoding-CKA: compare the representational geometry of
    # predicted responses to that of actual responses.  This is a hybrid metric
    # because the predictions already live in neural space (n_stimuli, n_units),
    # so RSA/CKA directly measures geometric fidelity of the encoding model.
    encoding_rsa = float(rsa(y_pred, y_true))
    encoding_cka = float(cka(y_pred, y_true))

    return {
        **_summarise(pearson,    "pearson_corr"),
        **_summarise(ev,         "explained_var"),
        **_summarise(nc_pearson, "noise_corrected_pearson"),
        **_summarise(nc_ev,      "noise_corrected_ev"),
        "noise_ceiling_mean": float(np.mean(noise_ceiling)),
        "encoding_rsa": encoding_rsa,
        "encoding_cka": encoding_cka,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(entry, base_dir, device):
    """
    Re-derive test-set predictions for one layer entry.

    Re-fits StandardScalers on the full training set (deterministic, same as
    the original SGDEncoder.fit call), then applies the saved linear weights.

    Returns
    -------
    y_test : np.ndarray, shape (n_stimuli, n_units)
    y_pred : np.ndarray, shape (n_stimuli, n_units)
    """
    model_name     = entry["model"]
    dataset_name   = entry["dataset"]
    roi            = entry["roi"]
    subject        = entry.get("subject")
    layer_name     = entry["layer"]
    neural_dataset = _neural_dataset_from_path(entry["weights_file"])

    if neural_dataset is None:
        raise ValueError("Cannot parse neural_dataset from weights_file path")

    # neural data
    y_train, stim_train = _load_neural(neural_dataset, roi, subject, "train")
    y_test,  stim_test  = _load_neural(neural_dataset, roi, subject, "test")

    # model features
    X_train = _load_features(model_name, dataset_name, stim_train, layer_name)
    X_test  = _load_features(model_name, dataset_name, stim_test,  layer_name)

    # re-fit scalers on full training set (same as SGDEncoder.fit)
    X_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    X_test_norm = X_scaler.transform(X_test)

    # load model weights
    weights_path = base_dir / entry["weights_file"]
    n_features = X_test_norm.shape[1]
    n_outputs  = y_test.shape[1] if y_test.ndim > 1 else 1

    model = LinearRegressionModel(n_features, n_outputs).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    with torch.no_grad():
        y_pred_norm = model(
            torch.FloatTensor(X_test_norm).to(device)
        ).cpu().numpy()

    y_pred = y_scaler.inverse_transform(y_pred_norm)

    if y_test.ndim == 1:
        y_test = y_test[:, None]
        y_pred = y_pred[:, None]

    return y_test, y_pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Add predictive and representational metrics to result JSONs."
    )
    parser.add_argument("--results-dir", default="./results",
                        help="Directory containing result JSON files (default: ./results)")
    parser.add_argument("--base-dir", default=".",
                        help="Base directory for resolving relative weights paths (default: .)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    base_dir    = Path(args.base_dir)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {results_dir}")
        sys.exit(0)

    print(f"Found {len(json_files)} JSON file(s)  |  device: {device}")

    # shared metric instances — instantiated once and reused
    rsa = RepresentationalSimilarityAnalysis()
    cka = CenteredKernelAlignment()

    for json_path in json_files:
        print(f"\n{json_path.name}")
        with open(json_path) as f:
            entries = json.load(f)

        updated = 0
        for entry in entries:
            if ADDED_KEYS.issubset(entry.keys()):
                continue

            layer = entry.get("layer", "?")
            print(f"  {layer} ... ", end="", flush=True)
            try:
                y_test, y_pred = run_inference(entry, base_dir, device)
                entry.update(compute_extra_metrics(y_test, y_pred, rsa, cka))
                updated += 1
                print("ok")
            except Exception as exc:
                print(f"FAILED — {exc}")

        if updated:
            with open(json_path, "w") as f:
                json.dump(entries, f, indent=2)
            print(f"  → updated {updated}/{len(entries)} entries")
        else:
            print("  → already up-to-date")


if __name__ == "__main__":
    main()
