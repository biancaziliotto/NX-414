#!/usr/bin/env python
"""
Rank-sweep encoding models on TVSD: train on the single best layer per
(model, ROI) with progressively decreasing low-rank bottleneck.

Ranks tried (in order):
  30000  — full-rank baseline, loaded directly from existing single-layer
            results (no re-training).
   3000  — low-rank, trained fresh with LowRankLinearModel (W = UV).
    300  — idem.
     30  — idem.

Alpha is taken from the cross-validated single-layer results, unchanged.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error

from utils.predictive_alignement import ModelBrainDataset, SGDEncoder
from utils.inspection_utils import load_tsvd_dataset
from utils.evaluation_metrics import compute_all_metrics
from utils.alignement_utils import RepresentationalSimilarityAnalysis, CenteredKernelAlignment


# ---------------------------------------------------------------------------
# Fixed configuration
# ---------------------------------------------------------------------------
MODELS = {
    "adv_resnet152_imagenet_full_ffgsm_eps-1_alpha-125-ep10_seed-0": "adv_resnet",
    "Qwen3-VL-2B-Instruct": "Qwen3-VL-2",
}
ROIS = ["V1", "V4", "IT"]
NEURAL_DATASET = "TVSD"
DATASET = "things_stimuli"
SUBJECT = "monkeyF"
# Ranks tried per (model, ROI). 30000 = full-rank baseline (loaded from existing
# results, not re-trained). n_units is substituted at runtime from the data.
FIXED_RANKS = [20, 10, 8, 6, 4]  # low-rank values, same for all ROIs


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_single_layer_results(results_dir: Path, model_alias: str, roi: str) -> list:
    path = results_dir / f"{model_alias}_things_stimuli_{NEURAL_DATASET}_{roi}__monkeyF_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Single-layer results not found: {path}")
    with open(path) as f:
        return json.load(f)


def best_layer_entry(single_results: list) -> dict:
    """Return the result dict for the layer with highest r2_mean."""
    return max(single_results, key=lambda r: r["r2_mean"])


# ---------------------------------------------------------------------------
# Training routine (low-rank only; full-rank comes from existing results)
# ---------------------------------------------------------------------------

def train_low_rank(
    model_name: str,
    dataset_name: str,
    roi: str,
    layer_name: str,
    fixed_alpha: float,
    rank: int,
    subject: str = SUBJECT,
    max_epochs: int = 1000,
    min_epochs: int = 20,
    patience: int = 10,
    tolerance: float = 1e-4,
    batch_size: int = 2048,
    learning_rate: float = 1e-4,
    verbose: bool = True,
) -> dict:
    """
    Train a single-layer low-rank encoder W = UV with a fixed rank and alpha.
    No hyperparameter search is performed.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  layer  : {layer_name}")
        print(f"  alpha  : {fixed_alpha}  rank: {rank}")

    timings: dict[str, float] = {}
    t_total = time.time()

    # ---- Neural data -------------------------------------------------------
    t0 = time.time()
    y_train, stimuli_train = load_tsvd_dataset(split="train", subject=subject, roi=roi)
    y_test, stimuli_test = load_tsvd_dataset(split="test", subject=subject, roi=roi)
    timings["data_loading"] = time.time() - t0

    # ---- Activations -------------------------------------------------------
    t0 = time.time()
    dataset = ModelBrainDataset(
        y_train=y_train, y_test=y_test,
        stimuli_train=stimuli_train, stimuli_test=stimuli_test,
        model_name=model_name, dataset_name=dataset_name,
        layer_name=layer_name,
    )
    timings["dataset_creation"] = time.time() - t0

    if verbose:
        print(f"  X_train {dataset.X_train.shape}  X_test {dataset.X_test.shape}")
        print(f"  y_train {dataset.y_train.shape}  y_test {dataset.y_test.shape}")

    # ---- Fit ---------------------------------------------------------------
    encoder = SGDEncoder(
        alpha=fixed_alpha,
        max_iter=max_epochs,
        min_iter=min_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=patience,
        early_stopping_tol=tolerance,
        random_state=42,
        rank=rank,
    )

    t0 = time.time()
    encoder.fit(dataset.X_train, dataset.y_train, verbose=False)
    y_pred_test  = encoder.predict(dataset.X_test)
    y_pred_train = encoder.predict(dataset.X_train)
    y_test_arr  = dataset.y_test
    y_train_arr = dataset.y_train
    timings["training_and_evaluation"] = time.time() - t0

    # ---- Weights -----------------------------------------------------------
    enc_dir = (
        Path("encoders") / model_name / dataset_name / NEURAL_DATASET / roi / subject
    )
    enc_dir.mkdir(parents=True, exist_ok=True)
    weights_file = enc_dir / f"{layer_name}_rank{rank}.pth"
    torch.save(encoder.model.state_dict(), weights_file)

    # ---- Metrics (test + train) --------------------------------------------
    t0 = time.time()
    rsa = RepresentationalSimilarityAnalysis(similarity_metric="pearson")
    cka = CenteredKernelAlignment()

    # Test
    r2_test  = [r2_score(y_test_arr[:, i], y_pred_test[:, i])  for i in range(y_test_arr.shape[1])]
    mse_test = [mean_squared_error(y_test_arr[:, i], y_pred_test[:, i]) for i in range(y_test_arr.shape[1])]
    metrics_test = compute_all_metrics(y_test_arr, y_pred_test)
    feature_rsa  = float(rsa(dataset.X_test, y_test_arr))
    feature_cka  = float(cka(dataset.X_test, y_test_arr))
    encoding_rsa = float(rsa(y_pred_test, y_test_arr))
    encoding_cka = float(cka(y_pred_test, y_test_arr))

    # Train
    r2_train  = [r2_score(y_train_arr[:, i], y_pred_train[:, i])  for i in range(y_train_arr.shape[1])]
    mse_train = [mean_squared_error(y_train_arr[:, i], y_pred_train[:, i]) for i in range(y_train_arr.shape[1])]
    metrics_train = compute_all_metrics(y_train_arr, y_pred_train)
    encoding_rsa_train = float(rsa(y_pred_train, y_train_arr))
    encoding_cka_train = float(cka(y_pred_train, y_train_arr))

    timings["metrics_calculation"] = time.time() - t0
    timings["total"] = time.time() - t_total

    result = {
        "rank": rank,
        "model_type": "low_rank",
        "layer": layer_name,
        "fixed_alpha": fixed_alpha,
        "model": model_name,
        "dataset": dataset_name,
        "roi": roi,
        "subject": subject,
        "X_train_shape": list(dataset.X_train.shape),
        "X_test_shape": list(dataset.X_test.shape),
        "y_train_shape": list(dataset.y_train.shape),
        "y_test_shape": list(dataset.y_test.shape),
        # test metrics
        "r2_mean": float(np.mean(r2_test)),
        "r2_std": float(np.std(r2_test)),
        "r2_median": float(np.median(r2_test)),
        "r2_min": float(np.min(r2_test)),
        "r2_max": float(np.max(r2_test)),
        "mse_mean": float(np.mean(mse_test)),
        "mse_std": float(np.std(mse_test)),
        **metrics_test,
        "feature_rsa": feature_rsa,
        "feature_cka": feature_cka,
        "encoding_rsa": encoding_rsa,
        "encoding_cka": encoding_cka,
        # train metrics (suffixed _train)
        "r2_mean_train": float(np.mean(r2_train)),
        "r2_std_train": float(np.std(r2_train)),
        "r2_median_train": float(np.median(r2_train)),
        "r2_min_train": float(np.min(r2_train)),
        "r2_max_train": float(np.max(r2_train)),
        "mse_mean_train": float(np.mean(mse_train)),
        "mse_std_train": float(np.std(mse_train)),
        **{k + "_train": v for k, v in metrics_train.items()},
        "encoding_rsa_train": encoding_rsa_train,
        "encoding_cka_train": encoding_cka_train,
        "n_units": len(r2_test),
        "timings": timings,
        "weights_file": str(weights_file),
    }

    if verbose:
        print(f"  {'':6s}  {'R²':>8s}  {'Pearson':>8s}  {'Expl.Var':>9s}  {'Enc.RSA':>8s}  {'Enc.CKA':>8s}")
        print(f"  {'TRAIN':6s}  {result['r2_mean_train']:8.4f}  "
              f"{result['pearson_corr_mean_train']:8.4f}  "
              f"{result['explained_var_mean_train']:9.4f}  "
              f"{result['encoding_rsa_train']:8.4f}  "
              f"{result['encoding_cka_train']:8.4f}")
        print(f"  {'TEST':6s}  {result['r2_mean']:8.4f}  "
              f"{result['pearson_corr_mean']:8.4f}  "
              f"{result['explained_var_mean']:9.4f}  "
              f"{result['encoding_rsa']:8.4f}  "
              f"{result['encoding_cka']:8.4f}")
        print(f"  ⏱  {timings['total']:.1f}s total")

    return result


def full_rank_entry(existing: dict) -> dict:
    """
    Wrap the existing single-layer result as the rank-30000 (full-rank) baseline.
    Adds normalised keys so it sits alongside the low-rank entries in the output JSON.
    """
    return {
        "rank": 30000,
        "model_type": "full_rank",
        "layer": existing["layer"],
        "fixed_alpha": existing["best_alpha"],
        **{k: existing[k] for k in (
            "model", "dataset", "roi", "subject",
            "X_train_shape", "X_test_shape", "y_train_shape", "y_test_shape",
            "r2_mean", "r2_std", "r2_median", "r2_min", "r2_max",
            "mse_mean", "mse_std",
            "pearson_corr_mean", "pearson_corr_std",
            "pearson_corr_median", "pearson_corr_min", "pearson_corr_max",
            "explained_var_mean", "explained_var_std",
            "explained_var_median", "explained_var_min", "explained_var_max",
            "noise_corrected_pearson_mean", "noise_corrected_pearson_std",
            "noise_corrected_pearson_median", "noise_corrected_pearson_min",
            "noise_corrected_pearson_max",
            "noise_corrected_ev_mean", "noise_corrected_ev_std",
            "noise_corrected_ev_median", "noise_corrected_ev_min",
            "noise_corrected_ev_max",
            "noise_ceiling_mean",
            "feature_rsa", "feature_cka", "encoding_rsa", "encoding_cka",
            "n_units", "timings", "weights_file",
        ) if k in existing},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rank-sweep encoding models on TVSD best layer per (model, ROI)."
    )
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-rank output")
    args = parser.parse_args()

    verbose = not args.quiet
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model_alias in MODELS.items():
        for roi in ROIS:
            print(f"\n{'#'*70}")
            print(f"  Model : {model_alias}  |  ROI : {roi}")
            print(f"{'#'*70}")

            try:
                single_results = load_single_layer_results(results_dir, model_alias, roi)
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}")
                continue

            best = best_layer_entry(single_results)
            best_layer = best["layer"]
            fixed_alpha = best["best_alpha"]
            ranks = [30000] + FIXED_RANKS

            print(f"  Best layer  : {best_layer}  (R²={best['r2_mean']:.4f})")
            print(f"  n_units     : {best['n_units']}")
            print(f"  Fixed alpha : {fixed_alpha}")
            print(f"  Ranks       : {ranks}  (rank=30000 loaded from existing results)")

            out_json = output_dir / f"{model_alias}_things_stimuli_TVSD_{roi}_rank_sweep_results.json"
            all_results: list[dict] = []

            for rank in ranks:
                if rank == 30000:
                    # Full-rank baseline — already trained, no re-training needed
                    entry = full_rank_entry(best)
                    if verbose:
                        print(f"\n  rank={rank:>6d}  [full-rank, loaded]  "
                              f"R²={entry['r2_mean']:.4f}  "
                              f"Pearson={entry['pearson_corr_mean']:.4f}")
                    all_results.append(entry)
                else:
                    try:
                        entry = train_low_rank(
                            model_name=model_name,
                            dataset_name=DATASET,
                            roi=roi,
                            layer_name=best_layer,
                            fixed_alpha=fixed_alpha,
                            rank=rank,
                            subject=SUBJECT,
                            max_epochs=args.max_epochs,
                            min_epochs=args.min_epochs,
                            patience=args.patience,
                            tolerance=args.tolerance,
                            batch_size=args.batch_size,
                            learning_rate=args.learning_rate,
                            verbose=verbose,
                        )
                    except Exception as exc:
                        print(f"  [ERROR] rank={rank}: {exc}", file=sys.stderr)
                        continue
                    all_results.append(entry)

                # Save progressively
                with open(out_json, "w") as f:
                    json.dump(all_results, f, indent=2, cls=_NumpyEncoder)

            print(f"\n  Saved {len(all_results)} entries → {out_json}")

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
