#!/usr/bin/env python
"""
Train multi-layer linear encoding models on the TVSD dataset.

For each (model, ROI) combination this script:
  1. Loads the existing single-layer results to rank layers by R² and retrieve
     the alpha that was already selected during cross-validation.
  2. Trains models with top-1, top-2, ..., top-N layers by concatenating their
     activations, using that fixed alpha — no additional HP search is performed.

The readout can be full-rank (--rank not set) or low-rank (--rank R).
Output filenames include the rank so different runs don't overwrite each other.

Supported models : adv_resnet, Qwen3-VL-2B-Instruct
ROIs             : V1, V4, IT
Dataset          : TVSD / things_stimuli / monkeyF
"""

import argparse
import gc
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


class _NumpyEncoder(json.JSONEncoder):
    """Serialize numpy scalars / arrays to plain Python types."""
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
    pattern = f"{model_alias}_things_stimuli_{NEURAL_DATASET}_{roi}__monkeyF_results.json"
    path = results_dir / pattern
    if not path.exists():
        raise FileNotFoundError(f"Single-layer results not found: {path}")
    with open(path) as f:
        return json.load(f)


def sorted_layers_by_r2(single_results: list) -> list[tuple[str, float]]:
    """Return (layer_name, best_alpha) pairs sorted best-to-worst by r2_mean."""
    ranked = sorted(single_results, key=lambda r: r["r2_mean"], reverse=True)
    return [(r["layer"], r["best_alpha"]) for r in ranked]


# ---------------------------------------------------------------------------
# Core training routine (no HP search — fixed alpha)
# ---------------------------------------------------------------------------

def train_fixed_alpha(
    model_name: str,
    dataset_name: str,
    roi: str,
    layer_names: list[str],
    fixed_alpha: float,
    rank: int | None,
    subject: str = SUBJECT,
    data_root: str | None = None,
    max_epochs: int = 1000,
    min_epochs: int = 20,
    patience: int = 10,
    tolerance: float = 1e-4,
    batch_size: int = 2048,
    learning_rate: float = 1e-4,
    verbose: bool = True,
) -> dict:
    """
    Train a linear encoder on concatenated activations of `layer_names`.
    rank=None → full-rank readout; rank=R → low-rank factorisation W=UV.
    """
    k = len(layer_names)
    layer_tag = f"top{k}_layers"
    layer_display = f"top-{k} [{'+'.join(layer_names)}]"

    if verbose:
        print(f"\n{'='*70}")
        print(f"  layers : {layer_display}")
        print(f"  alpha  : {fixed_alpha}  rank: {rank}")

    timings: dict[str, float] = {}
    t_total = time.time()

    # ---- Load neural data ------------------------------------------------
    t0 = time.time()
    y_train, stimuli_train = load_tsvd_dataset(split="train", subject=subject, roi=roi, data_root=data_root)
    y_test, stimuli_test = load_tsvd_dataset(split="test", subject=subject, roi=roi, data_root=data_root)
    timings["data_loading"] = time.time() - t0

    # ---- Build dataset (loads + concatenates activations) ----------------
    t0 = time.time()
    dataset = ModelBrainDataset(
        y_train=y_train, y_test=y_test,
        stimuli_train=stimuli_train, stimuli_test=stimuli_test,
        model_name=model_name, dataset_name=dataset_name,
        layer_name=layer_names,
        data_root=data_root,
    )
    timings["dataset_creation"] = time.time() - t0

    if verbose:
        print(f"  X_train {dataset.X_train.shape}  X_test {dataset.X_test.shape}")
        print(f"  y_train {dataset.y_train.shape}  y_test {dataset.y_test.shape}")

    # ---- Fit with fixed alpha and fixed rank (full training data, no split) --
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
    y_pred = encoder.predict(dataset.X_test)
    y_test_arr = dataset.y_test
    timings["training_and_evaluation"] = time.time() - t0

    # ---- Save model weights ---------------------------------------------
    enc_dir = (
        Path("encoders") / model_name / dataset_name / NEURAL_DATASET / roi / subject
    )
    enc_dir.mkdir(parents=True, exist_ok=True)
    weights_file = enc_dir / f"{layer_tag}.pth"
    torch.save(encoder.model.state_dict(), weights_file)

    # ---- Compute metrics ------------------------------------------------
    t0 = time.time()
    r2_list = [r2_score(y_test_arr[:, i], y_pred[:, i]) for i in range(y_test_arr.shape[1])]
    mse_list = [mean_squared_error(y_test_arr[:, i], y_pred[:, i]) for i in range(y_test_arr.shape[1])]
    all_metrics = compute_all_metrics(y_test_arr, y_pred)

    rsa = RepresentationalSimilarityAnalysis(similarity_metric="pearson")
    cka = CenteredKernelAlignment()
    X_test = dataset.X_test
    feature_rsa = float(rsa(X_test, y_test_arr))
    feature_cka = float(cka(X_test, y_test_arr))
    encoding_rsa = float(rsa(y_pred, y_test_arr))
    encoding_cka = float(cka(y_pred, y_test_arr))
    timings["metrics_calculation"] = time.time() - t0
    timings["total"] = time.time() - t_total

    result = {
        "n_layers": k,
        "layers": layer_names,
        "layer": layer_display,
        "fixed_alpha": fixed_alpha,
        "rank": rank,
        "model": model_name,
        "dataset": dataset_name,
        "roi": roi,
        "subject": subject,
        "X_train_shape": list(dataset.X_train.shape),
        "X_test_shape": list(dataset.X_test.shape),
        "y_train_shape": list(dataset.y_train.shape),
        "y_test_shape": list(dataset.y_test.shape),
        "r2_mean": float(np.mean(r2_list)),
        "r2_std": float(np.std(r2_list)),
        "r2_median": float(np.median(r2_list)),
        "r2_min": float(np.min(r2_list)),
        "r2_max": float(np.max(r2_list)),
        "mse_mean": float(np.mean(mse_list)),
        "mse_std": float(np.std(mse_list)),
        **all_metrics,
        "feature_rsa": feature_rsa,
        "feature_cka": feature_cka,
        "encoding_rsa": encoding_rsa,
        "encoding_cka": encoding_cka,
        "n_units": len(r2_list),
        "timings": timings,
        "weights_file": str(weights_file),
    }

    if verbose:
        print(f"  R²     : {result['r2_mean']:.4f} ± {result['r2_std']:.4f}")
        print(f"  Pearson: {result['pearson_corr_mean']:.4f} ± {result['pearson_corr_std']:.4f}")
        print(f"  ⏱  {timings['total']:.1f}s total")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train top-k multi-layer encoders using pre-selected alphas (TVSD)."
    )
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root directory containing data/ and extracted_features/ "
                             "(default: $NX414_DATA_ROOT or /shared/NX-414)")
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="Directory containing single-layer results JSONs (default: ./results)")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Directory for output JSONs (default: ./results)")
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--rank", type=int, default=None,
                        help="Low-rank bottleneck for W=UV. If not set, uses full-rank readout.")
    parser.add_argument("--max-layers", type=int, default=3,
                        help="Maximum number of top layers to combine (default: 10)")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-layer output")
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model_alias in MODELS.items():
        for roi in ROIS:
            print(f"\n{'#'*70}")
            print(f"  Model : {model_alias}")
            print(f"  ROI   : {roi}")
            print(f"{'#'*70}")

            # Load and rank single-layer results
            try:
                single_results = load_single_layer_results(results_dir, model_alias, roi)
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}")
                continue

            ranked = sorted_layers_by_r2(single_results)
            n_total = min(len(ranked), args.max_layers)

            # Fixed alpha = alpha of the best single layer
            fixed_alpha = ranked[0][1]

            rank_tag = "fullrank" if args.rank is None else f"rank{args.rank}"

            print(f"  Layers ranked by R² (using top {n_total}):")
            for pos, (layer, alpha) in enumerate(ranked[:n_total], 1):
                print(f"    {pos:2d}. {layer:<40s}  alpha={alpha}")
            print(f"  Fixed alpha : {fixed_alpha} (from best layer: {ranked[0][0]})")
            print(f"  Rank        : {args.rank if args.rank is not None else 'full-rank'}")

            out_json = output_dir / f"{model_alias}_things_stimuli_TVSD_{roi}_multilayer_{rank_tag}_results.json"
            all_results: list[dict] = []

            for k in range(1, n_total + 1):
                top_k = [layer for layer, _ in ranked[:k]]
                try:
                    result = train_fixed_alpha(
                        model_name=model_name,
                        dataset_name=DATASET,
                        roi=roi,
                        layer_names=top_k,
                        fixed_alpha=fixed_alpha,
                        rank=args.rank,
                        subject=SUBJECT,
                        data_root=args.data_root,
                        max_epochs=args.max_epochs,
                        min_epochs=args.min_epochs,
                        patience=args.patience,
                        tolerance=args.tolerance,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        verbose=verbose,
                    )
                except Exception as exc:
                    print(f"  [ERROR] k={k}: {exc}", file=sys.stderr)
                    continue

                all_results.append(result)

                # Save progressively after each k
                with open(out_json, "w") as f:
                    json.dump(all_results, f, indent=2, cls=_NumpyEncoder)

                # Release CPU memory and flush GPU cache between k values
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print(f"\n  Saved {len(all_results)} entries → {out_json}")

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
