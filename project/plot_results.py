#!/usr/bin/env python
"""
Generate all predictive-alignment plots from result JSONs.

Produces the following set of figures from the encoding-model metrics:

    r2_mean               – mean R² on the held-out test set
    pearson_corr_mean     – mean Pearson r on the test set
    encoding_rsa          – RSA between predicted and actual responses
    encoding_cka          – CKA between predicted and actual responses

Generates:
    2_4_{neural_dataset}_layerwise_{roi}.png          – Layer-wise performance curves
    2_4_{neural_dataset}_roi_{model}.png              – ROI hierarchy per model
    2_4_{neural_dataset}_model_comparison.png         – Best-layer model comparison
    2_5_{neural_dataset}_ranking_comparison_{roi}.png – Ranking comparison across metrics
    2_5_{neural_dataset}_ranking_agreement.png        – Best-layer agreement matrix

Run augment_results.py first to ensure all metrics are present.

Usage:
    python plot_results.py [--results-dir ./results] [--figures-dir ./figures]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from utils.predictive_plots import (
    load_predictive_results,
    best_layer_table,
    plot_layerwise,
    plot_roi_alignment,
    plot_model_comparison,
    DEFAULT_METRICS,
    METRIC_LABELS,
)
from utils.ranking_comparison import (
    plot_ranking_comparison,
    plot_ranking_agreement_matrix,
)


# Short display names for models whose full paths are unwieldy
MODEL_ALIASES = {
    "adv_resnet152_imagenet_full_ffgsm_eps-1_alpha-125-ep10_seed-0": "ResNet",
    "Qwen3-VL-2B-Instruct": "Qwen",
}

# ROI display order per neural dataset
ROI_ORDER = {
    "TVSD": ["V1", "V4", "IT"],
    "NSD":  ["V1v", "V2v", "V3v", "hV4", "FFA-1", "VWFA-1", "PPA", "OPA", "EBA"],
    "EEG2": ["occipital_parietal"],
}


def main():
    parser = argparse.ArgumentParser(
        description="Plot predictive alignment results from result JSONs."
    )
    parser.add_argument("--results-dir", default="./results",
                        help="Directory containing result JSON files (default: ./results)")
    parser.add_argument("--figures-dir", default="./figures",
                        help="Directory to save figures (default: ./figures)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load all results                                                     #
    # ------------------------------------------------------------------ #
    print("Loading results ...")
    df = load_predictive_results(
        results_dir=str(results_dir),
        model_aliases=MODEL_ALIASES,
    )

    if df.empty:
        print(f"No results found in {results_dir}")
        sys.exit(0)

    available_metrics = df["metric"].unique().tolist()
    metrics = [m for m in DEFAULT_METRICS if m in available_metrics]

    if not metrics:
        print("None of the default metrics are present in the results yet.")
        print("Run augment_results.py first to compute them.")
        sys.exit(0)

    print(f"Metrics available: {metrics}")
    print(f"Neural datasets:   {sorted(df['neural_dataset'].unique())}")
    print(f"Models:            {sorted(df['model'].unique())}")
    print()

    neural_datasets = sorted(df["neural_dataset"].unique())
    all_models      = sorted(df["model"].unique())

    # ------------------------------------------------------------------ #
    # Plots per neural dataset                                             #
    # ------------------------------------------------------------------ #
    for nd in neural_datasets:
        sub = df[df["neural_dataset"] == nd]
        targets   = ROI_ORDER.get(nd, sorted(sub["target"].unique()))
        # restrict to targets actually present in results
        targets   = [t for t in targets if t in sub["target"].unique()]
        nd_models = sorted(sub["model"].unique())

        print(f"=== {nd} (targets: {targets}) ===")

        # -- layer-wise curves per ROI (both models on same axes) -------
        for roi in targets:
            save = figures_dir / f"2_4_{nd}_layerwise_{roi}.png"
            print(f"  layerwise {roi} ...", end=" ")
            plot_layerwise(
                sub, target=roi, neural_dataset=nd,
                metrics=metrics,
                title_prefix=nd,
                save_path=str(save),
            )
            plt.close()
            print("saved")

        # -- ROI hierarchy per model ------------------------------------
        for model in nd_models:
            safe_model = model.replace("/", "-").replace(" ", "_")
            save = figures_dir / f"2_4_{nd}_roi_{safe_model}.png"
            print(f"  ROI alignment {model} ...", end=" ")
            plot_roi_alignment(
                sub, model=model, neural_dataset=nd,
                metrics=metrics,
                roi_order=targets,
                title_prefix=nd,
                save_path=str(save),
            )
            plt.close()
            print("saved")

        # -- best-layer model comparison --------------------------------
        save = figures_dir / f"2_4_{nd}_model_comparison.png"
        print(f"  model comparison ...", end=" ")
        plot_model_comparison(
            sub, neural_dataset=nd,
            targets=targets,
            metrics=metrics,
            title_prefix=nd,
            save_path=str(save),
        )
        plt.close()
        print("saved")

        print()

    # ------------------------------------------------------------------ #
    # Ranking comparison figures (2.5)                                     #
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("RANKING COMPARISON FIGURES")
    print("=" * 70)

    ranking_metrics = [m for m in DEFAULT_METRICS if m in available_metrics]

    for nd in neural_datasets:
        sub = df[df["neural_dataset"] == nd]
        targets = ROI_ORDER.get(nd, sorted(sub["target"].unique()))
        targets = [t for t in targets if t in sub["target"].unique()]
        
        print(f"=== {nd} (targets: {targets}) ===")
        
        # -- ranking comparison per ROI ---------------------------------
        for roi in targets:
            safe_roi = roi.replace("/", "-").replace(" ", "_")
            save = figures_dir / f"2_5_{nd}_ranking_comparison_{safe_roi}.png"
            print(f"  ranking comparison {roi} ...", end=" ")
            plot_ranking_comparison(
                sub, target=roi, neural_dataset=nd,
                metrics=ranking_metrics,
                title_prefix=nd,
                save_path=str(save),
            )
            plt.close()
            print("saved")
        
        # -- ranking agreement matrix -----------------------------------
        save = figures_dir / f"2_5_{nd}_ranking_agreement.png"
        print(f"  ranking agreement matrix ...", end=" ")
        plot_ranking_agreement_matrix(
            sub, neural_dataset=nd,
            metrics=ranking_metrics,
            title_prefix=nd,
            save_path=str(save),
        )
        plt.close()
        print("saved")
        
        print()

    # ------------------------------------------------------------------ #
    # Summary tables                                                       #
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("BEST-LAYER SUMMARY")
    print("=" * 70)
    best = best_layer_table(df)

    for nd in neural_datasets:
        print(f"\n--- {nd} ---")
        sub_best = best[best["neural_dataset"] == nd]
        for metric in metrics:
            print(f"\n  {METRIC_LABELS.get(metric, metric)}")
            print(
                sub_best[sub_best["metric"] == metric][
                    ["model", "target", "layer", "score"]
                ].to_string(index=False)
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
