"""
Plotting utilities for predictive alignment results.

Mirrors the structure of alignement_utils (plot_layerwise_alignment,
plot_roi_alignment, plot_model_comparison, best_layer_table) but operates on
per-layer encoding-model metrics stored in the result JSONs under ./results/.

DataFrame schema (output of load_predictive_results):
    model          – model name (short alias when available)
    neural_dataset – TVSD | EEG2 | NSD
    target         – ROI name (e.g. "IT", "V1v")
    layer          – layer name
    metric         – one of the METRICS keys
    score          – float

This DataFrame is directly compatible with all plotting functions below.
"""

import glob
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.alignement_utils import sort_layer_names

# ---------------------------------------------------------------------------
# Metrics supported / their display labels
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "r2_mean":                      "R² test (mean)",
    "pearson_corr_mean":            "Pearson r (mean)",
    "explained_var_mean":           "Expl. Var (mean)",
    "feature_rsa":                  "Feature RSA",
    "feature_cka":                  "Feature CKA",
    "encoding_rsa":                 "Encoding RSA",
    "encoding_cka":                 "Encoding CKA",
    "noise_corrected_pearson_mean": "NC Pearson r (mean)",
    "noise_corrected_ev_mean":      "NC Expl. Var (mean)",
}

# Metrics included by default in all multi-panel plots
DEFAULT_METRICS = ["r2_mean", "pearson_corr_mean", "encoding_rsa", "encoding_cka"]

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _neural_dataset_from_path(weights_file):
    parts = Path(weights_file).parts
    try:
        idx = list(parts).index("encoders")
        return parts[idx + 3]
    except (ValueError, IndexError):
        return "unknown"


def load_predictive_results(
    results_dir: str = "./results",
    metrics: list = None,
    model_aliases: dict = None,
) -> pd.DataFrame:
    """
    Load all result JSONs from results_dir into a tidy DataFrame.

    Parameters
    ----------
    results_dir : str
        Directory containing ``*_results.json`` files.
    metrics : list, optional
        Which metric keys to include. Defaults to all keys in METRIC_LABELS.
    model_aliases : dict, optional
        Map from full model name to short display name
        (e.g. ``{"adv_resnet152_...": "ResNet"}``).
        Unknown names are kept as-is.

    Returns
    -------
    pd.DataFrame
        Columns: [model, neural_dataset, target, layer, metric, score].
    """
    if metrics is None:
        metrics = list(METRIC_LABELS.keys())
    if model_aliases is None:
        model_aliases = {}

    rows = []
    for json_path in sorted(glob.glob(f"{results_dir}/*.json")):
        with open(json_path) as f:
            entries = json.load(f)
        for entry in entries:
            model          = model_aliases.get(entry["model"], entry["model"])
            neural_dataset = _neural_dataset_from_path(entry["weights_file"])
            target         = entry["roi"]
            layer          = entry["layer"]
            for metric in metrics:
                if metric in entry:
                    rows.append(dict(
                        model=model,
                        neural_dataset=neural_dataset,
                        target=target,
                        layer=layer,
                        metric=metric,
                        score=float(entry[metric]),
                    ))

    return pd.DataFrame(rows, columns=["model", "neural_dataset", "target",
                                       "layer", "metric", "score"])


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def best_layer_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Best-scoring layer per (model, neural_dataset, target, metric).

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_predictive_results.

    Returns
    -------
    pd.DataFrame
        Columns [model, neural_dataset, target, metric, layer, score].
    """
    idx = df.groupby(["model", "neural_dataset", "target", "metric"])["score"].idxmax()
    best = df.loc[idx, ["model", "neural_dataset", "target", "metric", "layer", "score"]]
    return best.sort_values(["neural_dataset", "metric", "model", "target"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Layer-wise curves  (mirrors plot_layerwise_alignment)
# ---------------------------------------------------------------------------

def plot_layerwise(
    df: pd.DataFrame,
    target: str,
    neural_dataset: str,
    metrics: list = None,
    title_prefix: str = "",
    save_path: str = None,
) -> plt.Figure:
    """
    Layer-wise predictive alignment scores for all models on a single target.

    One subplot per metric (default: r2_mean, pearson_corr_mean,
    encoding_rsa, encoding_cka) arranged in a 1×N or 2×2 grid.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_predictive_results.
    target : str
        ROI / brain region to plot (e.g. "IT").
    neural_dataset : str
        Neural dataset to filter on (e.g. "TVSD").
    metrics : list, optional
        Which metrics to show. Defaults to DEFAULT_METRICS.
    title_prefix : str
        Prepended to each subplot title.
    save_path : str, optional
        If provided, saves the figure here.
    """
    if metrics is None:
        metrics = [m for m in DEFAULT_METRICS if m in df["metric"].unique()]

    sub = df[(df["target"] == target) & (df["neural_dataset"] == neural_dataset)]
    models = sorted(sub["model"].unique())

    layer_orders = {
        m: sort_layer_names(sub[sub["model"] == m]["layer"].unique().tolist())
        for m in models
    }
    n_layers = max(len(lo) for lo in layer_orders.values())

    ncols = min(len(metrics), 2)
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, metrics):
        label = METRIC_LABELS.get(metric, metric)
        for model in models:
            lo = layer_orders[model]
            grp = sub[(sub["model"] == model) & (sub["metric"] == metric)]
            scores = grp.set_index("layer").reindex(lo)["score"].values
            ax.plot(range(len(lo)), scores, marker="o", label=model)
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels(
            layer_orders[models[0]] if len(models) == 1
            else [f"{i}" for i in range(n_layers)],
            rotation=45, ha="right", fontsize=7,
        )
        prefix = f"{title_prefix}, " if title_prefix else ""
        ax.set_title(f"{prefix}{label}  [{neural_dataset} / {target}]")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Score")
        ax.legend()

    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# ROI-hierarchy curves  (mirrors plot_roi_alignment)
# ---------------------------------------------------------------------------

def plot_roi_alignment(
    df: pd.DataFrame,
    model: str,
    neural_dataset: str,
    metrics: list = None,
    roi_order: list = None,
    title_prefix: str = "",
    save_path: str = None,
) -> plt.Figure:
    """
    Layer-wise predictive scores across brain ROIs for a single model.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_predictive_results.
    model : str
        Model name to plot.
    neural_dataset : str
        Neural dataset to filter on.
    metrics : list, optional
        Metrics to show. Defaults to DEFAULT_METRICS.
    roi_order : list, optional
        Explicit ROI ordering. Defaults to unique targets in df.
    title_prefix : str
    save_path : str, optional
    """
    if metrics is None:
        metrics = [m for m in DEFAULT_METRICS if m in df["metric"].unique()]

    sub = df[(df["model"] == model) & (df["neural_dataset"] == neural_dataset)]
    layer_order = sort_layer_names(sub["layer"].unique().tolist())
    if roi_order is None:
        roi_order = list(dict.fromkeys(sub["target"].tolist()))

    ncols = min(len(metrics), 2)
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, metrics):
        label = METRIC_LABELS.get(metric, metric)
        metric_sub = sub[sub["metric"] == metric]
        for roi in roi_order:
            grp = metric_sub[metric_sub["target"] == roi]
            scores = grp.set_index("layer").reindex(layer_order)["score"].values
            ax.plot(range(len(layer_order)), scores, marker="o", label=roi)
        ax.set_xticks(range(len(layer_order)))
        ax.set_xticklabels(layer_order, rotation=45, ha="right", fontsize=7)
        prefix = f"{title_prefix}, " if title_prefix else ""
        ax.set_title(f"{prefix}{model}, {label}  [{neural_dataset}]")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Score")
        ax.legend()

    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Best-layer model comparison  (mirrors plot_model_comparison)
# ---------------------------------------------------------------------------

def plot_model_comparison(
    df: pd.DataFrame,
    neural_dataset: str,
    targets: list = None,
    metrics: list = None,
    title_prefix: str = "",
    save_path: str = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing best-layer scores across models and ROIs.

    For each target the best-layer score is taken per model, giving a single
    value per (model, target) pair — the direct head-to-head comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_predictive_results.
    neural_dataset : str
        Neural dataset to filter on.
    targets : list, optional
        ROIs to include and their order. Defaults to all in df.
    metrics : list, optional
        Metrics to show. Defaults to DEFAULT_METRICS.
    title_prefix : str
    save_path : str, optional
    """
    if metrics is None:
        metrics = [m for m in DEFAULT_METRICS if m in df["metric"].unique()]

    sub = df[df["neural_dataset"] == neural_dataset]
    best = best_layer_table(sub)

    if targets is None:
        targets = list(dict.fromkeys(best["target"].tolist()))
    else:
        best = best[best["target"].isin(targets)]

    models = sorted(best["model"].unique())
    x = np.arange(len(targets))
    width = 0.8 / max(len(models), 1)

    ncols = min(len(metrics), 2)
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, metrics):
        label = METRIC_LABELS.get(metric, metric)
        metric_best = best[best["metric"] == metric]
        for i, model in enumerate(models):
            scores = []
            for t in targets:
                row = metric_best[(metric_best["model"] == model) & (metric_best["target"] == t)]
                scores.append(float(row["score"].values[0]) if len(row) else np.nan)
            offset = (i - (len(models) - 1) / 2) * width
            ax.bar(x + offset, scores, width, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels(targets)
        prefix = f"{title_prefix}, " if title_prefix else ""
        ax.set_title(f"{prefix}best-layer {label}  [{neural_dataset}]")
        ax.set_ylabel("Score")
        ax.legend()

    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
