"""
Ranking comparison utilities for comparing how different metrics rank models and layers.
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.alignement_utils import sort_layer_names
from utils.predictive_plots import METRIC_LABELS, best_layer_table, DEFAULT_METRICS


def plot_ranking_comparison(
    df: pd.DataFrame,
    neural_dataset: str,
    target: str,
    model: str = None,
    metrics: list = None,
    title_prefix: str = "",
    save_path: str = None,
) -> plt.Figure:
    """
    Compare how different metrics rank layers for a given target/dataset.
    
    Shows best layers (bars) and layer-wise scores (curves) for each metric,
    enabling visual comparison of which layers are favored by different metrics
    and identification of agreements/disagreements.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_predictive_results.
    neural_dataset : str
        Neural dataset (e.g., "TVSD", "NSD").
    target : str
        ROI / brain region (e.g., "IT", "V1v").
    model : str, optional
        If provided, restrict to a single model. If None, include all available.
    metrics : list, optional
        Which metrics to compare. Defaults to DEFAULT_METRICS.
    title_prefix : str
        Prepended to the figure title.
    save_path : str, optional
        If provided, saves the figure here.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if metrics is None:
        metrics = [m for m in DEFAULT_METRICS if m in df["metric"].unique()]

    sub = df[(df["neural_dataset"] == neural_dataset) & (df["target"] == target)]
    
    if model is not None:
        sub = sub[sub["model"] == model]
    
    if sub.empty:
        print(f"No data found for {neural_dataset} / {target} / {model}")
        return None
    
    # Get unique models and layers
    models_list = sorted(sub["model"].unique())
    
    # Create one ax per metric
    ncols = min(len(metrics), 2)
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, metrics):
        metric_data = sub[sub["metric"] == metric]
        
        if metric_data.empty:
            ax.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center",
                   transform=ax.transAxes)
            continue
        
        # Plot curves for each model
        for model_name in models_list:
            model_data = metric_data[metric_data["model"] == model_name]
            layers = sort_layer_names(model_data["layer"].unique().tolist())
            scores = model_data.set_index("layer").reindex(layers)["score"].values
            ax.plot(range(len(layers)), scores, marker="o", label=model_name, linewidth=2)
            
            # Highlight best layer with a vertical line
            best_idx = np.nanargmax(scores)
            ax.axvline(best_idx, alpha=0.2, linestyle="--")
        
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Score")
        
        label = METRIC_LABELS.get(metric, metric)
        title = f"{label}  [{neural_dataset} / {target}]"
        if title_prefix:
            title = f"{title_prefix}, " + title
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ranking_agreement_matrix(
    df: pd.DataFrame,
    neural_dataset: str,
    metrics: list = None,
    title_prefix: str = "",
    save_path: str = None,
) -> plt.Figure:
    """
    Create a heatmap comparing best-layer agreement across metrics and targets.
    
    Shows for each (target, model) pair, the best layer according to each metric,
    making it easy to spot agreements (same layer across metrics) and disagreements.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_predictive_results.
    neural_dataset : str
        Neural dataset (e.g., "TVSD", "NSD").
    metrics : list, optional
        Which metrics to include. Defaults to DEFAULT_METRICS.
    title_prefix : str
        Prepended to the figure title.
    save_path : str, optional
        If provided, saves the figure here.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if metrics is None:
        metrics = [m for m in DEFAULT_METRICS if m in df["metric"].unique()]
    
    sub = df[df["neural_dataset"] == neural_dataset]
    best = best_layer_table(sub)
    
    targets = sorted(sub["target"].unique())
    models = sorted(sub["model"].unique())
    
    # For each (target, model) pair, collect the best layers per metric
    layer_map = {}  # (target, model) -> {metric -> layer}
    
    for target in targets:
        for model in models:
            layer_map[(target, model)] = {}
            for metric in metrics:
                row = best[(best["target"] == target) & 
                          (best["model"] == model) & 
                          (best["metric"] == metric)]
                if len(row) > 0:
                    layer_map[(target, model)][metric] = row["layer"].values[0]
    
    # Create figure with subplots per model
    ncols = min(len(models), 2)
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    
    for ax, model in zip(axes_flat, models):
        # Build data matrix: rows = targets, cols = metrics
        data = []
        for target in targets:
            row_data = []
            for metric in metrics:
                layer_str = layer_map.get((target, model), {}).get(metric, "N/A")
                row_data.append(layer_str)
            data.append(row_data)
        
        # Create table
        table = ax.table(cellText=data, 
                        rowLabels=targets,
                        colLabels=[METRIC_LABELS.get(m, m) for m in metrics],
                        cellLoc="center",
                        loc="center",
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Color cells to highlight agreements/disagreements
        for i, target in enumerate(targets):
            layers_for_target = [layer_map[(target, model)][m] for m in metrics 
                                if m in layer_map[(target, model)]]
            # Check if all layers are the same
            all_same = len(set(layers_for_target)) == 1 if layers_for_target else False
            
            for j in range(len(metrics)):
                cell = table[(i+1, j)]  # +1 because row 0 is header
                if all_same:
                    cell.set_facecolor("#90EE90")  # Light green for agreement
                else:
                    cell.set_facecolor("#FFB6C1")  # Light red for disagreement
        
        ax.axis("off")
        title = f"{model}  [{neural_dataset}]"
        if title_prefix:
            title = f"{title_prefix}, " + title
        ax.set_title(title, fontsize=11, fontweight="bold", pad=20)
    
    for ax in axes_flat[len(models):]:
        ax.set_visible(False)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
