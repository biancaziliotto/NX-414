import math
import numpy as np
from typing import Literal
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import re

class RepresentationalSimilarityAnalysis:
    """
    Representational Similarity Analysis (RSA).

    Given two representation matrices X and Y with the same number of conditions
    (rows), RSA:

    1. Computes a Representational Dissimilarity Matrix (RDM) for each:
       RDM_X[i, j] = dissimilarity(x_i, x_j)
       RDM_Y[i, j] = dissimilarity(y_i, y_j)

    2. Flattens the upper triangles of both RDMs and computes a correlation
       between them (Pearson or Spearman).
    """

    def __init__(
        self,
        dissimilarity: Literal["correlation", "euclidean", "cosine"] = "correlation",
        similarity_metric: Literal["pearson", "spearman"] = "spearman",
    ):
        ### TODO
        self.dissimilarity = dissimilarity
        self.similarity_metric = similarity_metric

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute RSA similarity between X and Y.

        Parameters
        ----------
        X, Y : np.ndarray
            Arrays of shape (n_conditions, ...) that may need to be flattened
            along feature dimensions.

        Returns
        -------
        rsa_similarity : float
            Correlation between the vectorized upper triangles of the two RDMs.
        """
        return self.forward(X, Y)

    def forward(self, X: np.ndarray, Y: np.ndarray) -> float:
        ### TODO
        rdm_X = self.compute_rdm(X)
        rdm_Y = self.compute_rdm(Y)
        return self.compare_rdms(rdm_X, rdm_Y)

    def compute_rdm(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the Representational Dissimilarity Matrix (RDM)
        for a given representation matrix X.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_conditions, n_features).

        Returns
        -------
        rdm : np.ndarray
            Array of shape (n_conditions, n_conditions) with pairwise dissimilarities.
        """
        ### TODO
        if self.dissimilarity == "correlation":
            # Compute correlation-based dissimilarity
            X_centered = X - X.mean(axis=0)
            rdm = 1 - np.corrcoef(X_centered)
        elif self.dissimilarity == "euclidean":
            # Compute Euclidean distance-based dissimilarity
            rdm = np.sqrt(((X[:, np.newaxis] - X) ** 2).sum(axis=2))
        elif self.dissimilarity == "cosine":
            # Compute cosine distance-based dissimilarity
            X_norm = np.linalg.norm(X, axis=1, keepdims=True)
            X_normalized = X / X_norm
            rdm = 1 - np.dot(X_normalized, X_normalized.T)
        else:
            raise ValueError(f"Unknown dissimilarity metric: {self.dissimilarity}")
        return rdm

    def compare_rdms(self, rdm1: np.ndarray, rdm2: np.ndarray) -> float:
        """
        Compare two RDMs by correlating their upper triangles.
        """
        ### TODO
        # Extract upper triangles
        triu_indices = np.triu_indices_from(rdm1, k=1)
        rdm1_vec = rdm1[triu_indices]
        rdm2_vec = rdm2[triu_indices]
        if self.similarity_metric == "pearson":
            # Compute Pearson correlation
            rdm1_centered = rdm1_vec - rdm1_vec.mean()
            rdm2_centered = rdm2_vec - rdm2_vec.mean()
            numerator = np.sum(rdm1_centered * rdm2_centered)
            denominator = np.sqrt(np.sum(rdm1_centered ** 2) * np.sum(rdm2_centered ** 2))
            return numerator / denominator if denominator != 0 else 0.0
        elif self.similarity_metric == "spearman":
            # Compute Spearman correlation
            rdm1_ranked = np.argsort(np.argsort(rdm1_vec))
            rdm2_ranked = np.argsort(np.argsort(rdm2_vec))
            rdm1_centered = rdm1_ranked - rdm1_ranked.mean()
            rdm2_centered = rdm2_ranked - rdm2_ranked.mean()
            numerator = np.sum(rdm1_centered * rdm2_centered)
            denominator = np.sqrt(np.sum(rdm1_centered ** 2) * np.sum(rdm2_centered ** 2))
            return numerator / denominator if denominator != 0 else 0.0

class CenteredKernelAlignment:
    """
    Unbiased linear CKA only.

    Parameters
    ----------
    eps : float
        Small constant for numerical stability.
    dtype : np.dtype
        Data type used for computations.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        dtype: np.dtype = np.float64,
    ):
        ### TODO
        self.eps = eps
        self.dtype = dtype

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> float:
        return self.forward(X, Y)

    def forward(self, X: np.ndarray, Y: np.ndarray) -> float:
        X = np.asarray(X).astype(self.dtype)
        Y = np.asarray(Y).astype(self.dtype)

        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Batch sizes must match along axis 0: {X.shape[0]} vs {Y.shape[0]}"
            )

        # Flatten to (n_samples, n_features)
        X = X.reshape(X.shape[0], -1)
        Y = Y.reshape(Y.shape[0], -1)

        return self._unbiased_linear_cka(X, Y)

    def _unbiased_linear_hsic(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Unbiased HSIC estimator for the linear kernel.

        X : [n, d_x]
        Y : [n, d_y]
        """
        n = X.shape[0]

        K = X @ X.T  # (n, n)
        L = Y @ Y.T  # (n, n)

        # Zero diagonals for the unbiased estimator
        np.fill_diagonal(K, 0)
        np.fill_diagonal(L, 0)

        # Unbiased HSIC U-statistic 
        term1 = np.einsum("ij,ji->", K, L)  # tr(K @ L) 
        sum_K = K.sum()
        sum_L = L.sum()
        term2 = (sum_K * sum_L) / ((n - 1) * (n - 2))
        term3 = (2.0 / (n - 2)) * np.dot(K.sum(axis=1), L.sum(axis=1))

        return (term1 + term2 - term3) / (n * (n - 3))


    def _unbiased_linear_cka(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Unbiased linear CKA:

            CKA_unb(X, Y) =
                HSIC_unb(X, Y) / sqrt(HSIC_unb(X, X) * HSIC_unb(Y, Y))
        """
        hsic_xy = self._unbiased_linear_hsic(X, Y)
        hsic_xx = self._unbiased_linear_hsic(X, X)
        hsic_yy = self._unbiased_linear_hsic(Y, Y)
        return hsic_xy / np.sqrt(hsic_xx * hsic_yy + self.eps)


# Layer-wise and cross-target scoring utilities

def compute_layer_scores(
    layer_features: dict,
    neural_responses: np.ndarray,
    rsa: RepresentationalSimilarityAnalysis = None,
    cka: CenteredKernelAlignment = None,
) -> dict:
    """
    Compute RSA and CKA for every layer against a single neural response matrix.

    Parameters
    ----------
    layer_features : dict
        {layer_name: np.ndarray of shape (n_stimuli, n_features)}.
        Layer order is preserved (use an OrderedDict or Python 3.7+ dict).
    neural_responses : np.ndarray
        Shape (n_stimuli, ...). Extra dims are flattened automatically.
        For EEG pass a single time slice (n_stimuli, n_channels) or the full
        flattened array (n_stimuli, n_channels * n_timepoints).
    rsa, cka : metric instances, optional
        Re-use pre-built instances to avoid re-initialising defaults each call.

    Returns
    -------
    dict
        {'RSA': {layer_name: score}, 'CKA': {layer_name: score}}
    """
    if rsa is None:
        rsa = RepresentationalSimilarityAnalysis()
    if cka is None:
        cka = CenteredKernelAlignment()

    Y = neural_responses.reshape(neural_responses.shape[0], -1)

    rsa_scores, cka_scores = {}, {}
    for layer_name, features in layer_features.items():
        X = np.asarray(features).reshape(features.shape[0], -1)
        rsa_scores[layer_name] = rsa(X, Y)
        cka_scores[layer_name] = cka(X, Y)

    return {"RSA": rsa_scores, "CKA": cka_scores}


def compare_models_and_targets(
    models_features: dict,
    target_responses: dict,
    rsa: RepresentationalSimilarityAnalysis = None,
    cka: CenteredKernelAlignment = None,
) -> dict:
    """
    Sweep all combinations of (model, target) and return layer-wise scores.

    Parameters
    ----------
    models_features : dict
        {model_name: {layer_name: np.ndarray (n_stimuli, n_features)}}
    target_responses : dict
        {target_name: np.ndarray (n_stimuli, ...)}.
        Targets can be ROIs, subjects, or EEG time slices, anything with
        a matching first axis.

    Returns
    -------
    dict
        Nested as results[model_name][target_name] = {'RSA': {...}, 'CKA': {...}}
    """
    if rsa is None:
        rsa = RepresentationalSimilarityAnalysis()
    if cka is None:
        cka = CenteredKernelAlignment()

    results = {}
    for model_name, layer_features in models_features.items():
        results[model_name] = {}
        for target_name, responses in target_responses.items():
            results[model_name][target_name] = compute_layer_scores(
                layer_features, responses, rsa=rsa, cka=cka
            )
    return results


def scores_to_dataframe(results: dict):
    """
    Flatten the nested results dict from compare_models_and_targets into a
    tidy DataFrame with columns [model, target, layer, metric, score].
    """
    rows = []
    for model, targets in results.items():
        for target, metrics in targets.items():
            for metric, layer_scores in metrics.items():
                for layer, score in layer_scores.items():
                    rows.append(
                        dict(model=model, target=target, layer=layer,
                             metric=metric, score=score)
                    )
    return pd.DataFrame(rows)


def sort_layer_names(layer_names: list) -> list:
    """
    Sort layer names in architectural order (early → late).

    Handles ResNet layers ("layer1-0", "layer3-10", …) and Qwen layers
    ("visual-blocks-2", "language_model-layers-8", …).
    For unknown patterns, falls back to lexicographic sort.
    """
    def sort_key(name):
        # ResNet: "layerN-M"  →  (N, M)
        m = re.fullmatch(r"layer(\d+)-(\d+)", name)
        if m:
            return (0, int(m.group(1)), int(m.group(2)))
        # Qwen visual encoder comes before language model
        m = re.fullmatch(r"visual-blocks-(\d+)", name)
        if m:
            return (1, 0, int(m.group(1)))
        m = re.fullmatch(r"language_model-layers-(\d+)", name)
        if m:
            return (1, 1, int(m.group(1)))
        # fallback: lexicographic
        return (2, 0, name)

    return sorted(layer_names, key=sort_key)


def plot_layerwise_alignment(
    df,
    target: str,
    title_prefix: str = "",
    save_path: str = None,
):
    """
    Plot RSA and CKA layer-wise brain-model alignment scores for two models.

    Produces 2 subplots (RSA, CKA). Both models are plotted on the same x-axis
    since they have the same number of layers. Each tick label shows both layer
    names stacked: "<Qwen layer>\n<ResNet layer>", so the positional correspondence
    is clear without mixing architecturally incomparable names on a single axis label.

    Parameters
    ----------
    df : pd.DataFrame
        Output of scores_to_dataframe; columns [model, target, layer, metric, score].
    target : str
        Brain target to filter on (e.g. "IT").
    title_prefix : str
        Prepended to each subplot title.
    save_path : str, optional
        If provided, saves the figure to this path (should end with .png).
    """
    sub = df[df.target == target]
    models = sorted(sub["model"].unique())  # deterministic order

    # sort each model's layers independently in architectural order
    layer_orders = {
        model: sort_layer_names(sub[sub.model == model]["layer"].unique().tolist())
        for model in models
    }

    # build dual tick labels: "qwen_layer\nresnet_layer" at each position
    n_layers = len(layer_orders[models[0]])
    tick_labels = [
        "\n".join(layer_orders[m][i] for m in models)
        for i in range(n_layers)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ["RSA", "CKA"]):
        for model in models:
            grp = sub[(sub.model == model) & (sub.metric == metric)]
            scores = grp.set_index("layer").reindex(layer_orders[model])["score"].values
            ax.plot(range(n_layers), scores, marker="o", label=model)
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        prefix = f"{title_prefix}, " if title_prefix else ""
        ax.set_title(f"{prefix}{metric}, {target}")
        ax.set_xlabel("Layer index  (Qwen / ResNet)")
        ax.set_ylabel("Score")
        ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roi_alignment(
    df,
    model: str,
    roi_order: list = None,
    layer_order: list = None,
    title_prefix: str = "",
    save_path: str = None,
):
    """
    Plot RSA and CKA layer-wise scores across brain ROIs for a single model.

    Parameters
    ----------
    df : pd.DataFrame
        Output of scores_to_dataframe; columns [model, target, layer, metric, score].
    model : str
        Model name to plot (e.g. "ResNet").
    roi_order : list, optional
        Explicit ROI ordering; defaults to unique targets in df.
    layer_order : list, optional
        Explicit layer ordering; defaults to the order found in df.
    title_prefix : str
        Prepended to each subplot title.
    save_path : str, optional
        If provided, saves the figure to this path (should end with .png).
    """
    sub = df[df.model == model]
    if layer_order is None:
        layer_order = sort_layer_names(sub["layer"].unique().tolist())  # architectural order
    if roi_order is None:
        roi_order = list(dict.fromkeys(sub["target"].tolist()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ["RSA", "CKA"]):
        metric_sub = sub[sub.metric == metric]
        for roi in roi_order:
            grp = metric_sub[metric_sub.target == roi]
            scores = grp.set_index("layer").reindex(layer_order)["score"].values
            ax.plot(range(len(layer_order)), scores, marker="o", label=roi)
        ax.set_xticks(range(len(layer_order)))
        ax.set_xticklabels(layer_order, rotation=45, ha="right")
        title = f"{title_prefix}, " if title_prefix else ""
        ax.set_title(f"{title}{model}, {metric} across ROIs")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Score")
        ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def best_layer_table(df):
    """
    Return a summary table of the best-scoring layer per (model, target, metric).

    Parameters
    ----------
    df : pd.DataFrame
        Output of scores_to_dataframe.

    Returns
    -------
    pd.DataFrame
        Columns [model, target, metric, layer, score], one row per combination.
    """
    idx = df.groupby(["model", "target", "metric"])["score"].idxmax()
    best = df.loc[idx, ["model", "target", "metric", "layer", "score"]]
    return best.sort_values(["metric", "model", "target"]).reset_index(drop=True)


def plot_model_comparison(
    df: pd.DataFrame,
    targets: list = None,
    title_prefix: str = "",
    save_path: str = None,
):
    """
    Grouped bar chart comparing best-layer RSA and CKA scores between models.

    For each target (ROI / dataset), the best-layer score is taken per model,
    so each model is summarised by a single value per target rather than a
    curve over layers. This is the direct head-to-head comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Output of scores_to_dataframe; columns [model, target, layer, metric, score].
    targets : list, optional
        Targets to include and their plotting order. Defaults to all targets in df.
    title_prefix : str
        Prepended to subplot titles.
    save_path : str, optional
        If provided, saves the figure to this path.
    """
    best = best_layer_table(df)
    if targets is None:
        targets = list(dict.fromkeys(best["target"].tolist()))
    else:
        best = best[best["target"].isin(targets)]

    models = sorted(best["model"].unique())
    x = np.arange(len(targets))
    width = 0.8 / max(len(models), 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ["RSA", "CKA"]):
        sub = best[best["metric"] == metric]
        for i, model in enumerate(models):
            scores = []
            for t in targets:
                row = sub[(sub["model"] == model) & (sub["target"] == t)]
                scores.append(row["score"].values[0] if len(row) else np.nan)
            offset = (i - (len(models) - 1) / 2) * width
            ax.bar(x + offset, scores, width, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels(targets)
        prefix = f"{title_prefix}, " if title_prefix else ""
        ax.set_title(f"{prefix}best-layer {metric} per target")
        ax.set_ylabel("Score")
        ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def compute_eeg_timeresolved(
    models_features: dict,
    eeg_data: np.ndarray,
    rsa: RepresentationalSimilarityAnalysis = None,
    cka: CenteredKernelAlignment = None,
) -> dict:
    """
    Best-layer RSA and CKA scores across EEG time points, per model.

    For each time point t, slices the EEG response matrix
    Y_t = eeg_data[:, :, t] of shape (n_stimuli, n_channels), runs RSA and CKA
    against every layer of each model, and keeps the maximum score.

    Parameters
    ----------
    models_features : dict
        {model_name: {layer_name: np.ndarray (n_stimuli, n_features)}}
    eeg_data : np.ndarray
        Shape (n_stimuli, n_channels, n_timepoints).
    rsa, cka : metric instances, optional
        Pre-built instances; defaults are created if missing.

    Returns
    -------
    dict
        {model_name: {'RSA': np.ndarray (n_timepoints,),
                      'CKA': np.ndarray (n_timepoints,)}}
    """
    if rsa is None:
        rsa = RepresentationalSimilarityAnalysis()
    if cka is None:
        cka = CenteredKernelAlignment()

    n_timepoints = eeg_data.shape[2]
    results = {}

    for model_name, layer_features in models_features.items():
        rsa_best = np.empty(n_timepoints)
        cka_best = np.empty(n_timepoints)
        for t in range(n_timepoints):
            Y_t = eeg_data[:, :, t]
            scores = compute_layer_scores(layer_features, Y_t, rsa=rsa, cka=cka)
            rsa_best[t] = max(scores["RSA"].values())
            cka_best[t] = max(scores["CKA"].values())
        results[model_name] = {"RSA": rsa_best, "CKA": cka_best}

    return results


def plot_eeg_timeresolved(
    time_scores: dict,
    time_ms: np.ndarray,
    title_prefix: str = "",
    save_path: str = None,
):
    """
    Plot best-layer RSA and CKA over EEG time, with both models on the same axes.

    Parameters
    ----------
    time_scores : dict
        Output of compute_eeg_timeresolved.
    time_ms : np.ndarray
        Time axis in milliseconds, length n_timepoints.
    title_prefix : str
        Prepended to subplot titles.
    save_path : str, optional
        If provided, saves the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ["RSA", "CKA"]):
        for model_name, scores in time_scores.items():
            ax.plot(time_ms, scores[metric], marker="", label=model_name)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Time after stimulus onset (ms)")
        ax.set_ylabel(f"Best-layer {metric}")
        prefix = f"{title_prefix}, " if title_prefix else ""
        ax.set_title(f"{prefix}EEG time-resolved {metric}")
        ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def load_eeg_test(eeg_path: str, subject: str = "sub-01", region: str = "temporal"):
    """
    Load EEG test responses and stimulus IDs for one (subject, region).

    Returns
    -------
    eeg_data : np.ndarray, shape (n_stimuli, n_channels, n_timepoints)
    eeg_ids  : np.ndarray, shape (n_stimuli,) — stimulus identifiers
    """
    with h5py.File(eeg_path, "r") as f:
        eeg_ids  = f["test/stimulus_ids"][:]
        eeg_data = f[f"test/neural_data/{subject}/{region}"][:]
    return eeg_data, eeg_ids


def load_nsd_test(nsd_path: str, subject: str = "subj01", rois: list = None):
    """
    Load NSD test responses and stimulus IDs for one subject across several ROIs.

    Parameters
    ----------
    nsd_path : str
        Path to nsd_func1pt8mm_individualROIs.h5
    subject : str
        Subject key (e.g. "subj01").
    rois : list, optional
        ROI names. Defaults to early-/mid-/high-level ventral stream:
        ["V1v", "hV4", "ventral"].

    Returns
    -------
    nsd_rois : dict
        {roi_name: np.ndarray (n_stimuli, n_voxels)}
    nsd_ids : np.ndarray, shape (n_stimuli,)
        Subject-specific NSD integer stimulus IDs.
    """
    if rois is None:
        rois = ["V1v", "V2v", "V3v", "hV4", "FFA-1", "VWFA-1", "PPA", "OPA", "EBA"]
    with h5py.File(nsd_path, "r") as f:
        nsd_ids  = f[f"test/stimulus_ids/{subject}"][:]
        nsd_rois = {roi: f[f"test/neural_data/{subject}/{roi}"][:] for roi in rois}
    return nsd_rois, nsd_ids


# Bundled plotting

def plot_alignment_full(
    df: pd.DataFrame,
    targets: list,
    title_prefix: str,
    save_dir: str = "figures",
    file_prefix: str = "alignment",
    show_layerwise: bool = True,
    show_model_comparison: bool = True,
    show_roi_hierarchy: bool = True,
    print_best: bool = True,
):
    """
    Run the full set of representational-alignment plots from a scores DataFrame.

    Produces, in order:
      1. Layer-wise alignment plot (both models on same axes) for each target.
      2. Grouped bar chart comparing the two models' best-layer scores per target.
      3. Per-model ROI hierarchy plot (one figure per model).
      4. Prints the best-layer summary table.

    All figures are saved under ``{save_dir}/{file_prefix}_*.png``.

    Parameters
    ----------
    df : pd.DataFrame
        Output of scores_to_dataframe (columns [model, target, layer, metric, score]).
    targets : list
        Targets to include and their plotting order (e.g. ["V1", "V4", "IT"]).
    title_prefix : str
        Title prefix shared across plots (e.g. "TVSD monkeyF" or "NSD avg").
    save_dir : str
        Directory where figures are written. Must already exist.
    file_prefix : str
        Filename prefix (e.g. "2_3_tvsd").
    show_layerwise, show_model_comparison, show_roi_hierarchy, print_best : bool
        Toggles for individual plots and the summary print.
    """
    if show_layerwise:
        for roi in targets:
            plot_layerwise_alignment(
                df, target=roi, title_prefix=title_prefix,
                save_path=f"{save_dir}/{file_prefix}_layerwise_{roi}.png",
            )
            plt.show()

    if show_model_comparison:
        plot_model_comparison(
            df, targets=targets, title_prefix=title_prefix,
            save_path=f"{save_dir}/{file_prefix}_model_comparison.png",
        )
        plt.show()

    if show_roi_hierarchy:
        for model in sorted(df["model"].unique()):
            plot_roi_alignment(
                df, model=model, roi_order=targets,
                title_prefix=title_prefix,
                save_path=f"{save_dir}/{file_prefix}_roi_{model}.png",
            )
            plt.show()

    if print_best:
        print(f"=== {title_prefix} best layers ===")
        print(best_layer_table(df).to_string(index=False))


# Multi-subject helpers

def average_scores_df(per_subject_dfs: dict) -> pd.DataFrame:
    """
    Average alignment scores across subjects.

    Parameters
    ----------
    per_subject_dfs : dict
        {subject_name: DataFrame} where each DataFrame has columns
        [model, target, layer, metric, score] (output of scores_to_dataframe).

    Returns
    -------
    pd.DataFrame
        Same columns, scores averaged across subjects.
    """
    combined = pd.concat(
        [df.assign(subject=sub) for sub, df in per_subject_dfs.items()],
        ignore_index=True,
    )
    return (
        combined
        .groupby(["model", "target", "layer", "metric"], as_index=False)["score"]
        .mean()
    )


def average_timeresolved(per_subject_scores: dict) -> dict:
    """
    Average EEG time-resolved scores across subjects.

    Parameters
    ----------
    per_subject_scores : dict
        {subject: {model: {'RSA': array, 'CKA': array}}}

    Returns
    -------
    dict
        {model: {'RSA': array, 'CKA': array}} averaged across subjects.
    """
    subjects = list(per_subject_scores.keys())
    models   = list(per_subject_scores[subjects[0]].keys())
    return {
        model: {
            metric: np.mean(
                [per_subject_scores[s][model][metric] for s in subjects], axis=0
            )
            for metric in ("RSA", "CKA")
        }
        for model in models
    }


def tvsd_alignment_multisubject(
    tvsd_path: str,
    rois: list,
    resnet_path: str,
    qwen_path: str,
    monkeys: list = None,
    rsa: RepresentationalSimilarityAnalysis = None,
    cka: CenteredKernelAlignment = None,
):
    """
    Run RSA/CKA alignment for each TVSD monkey across the given ROIs.

    TVSD test stimulus IDs are shared across monkeys, so model features are
    loaded once and reused.

    Returns
    -------
    per_monkey : dict {monkey: DataFrame}
    df_avg     : DataFrame averaged across monkeys
    """
    if monkeys is None:
        monkeys = ["monkeyF", "monkeyN"]
    with h5py.File(tvsd_path, "r") as f:
        ids = f["test/stimulus_ids"][:]
    resnet_layers = load_features(resnet_path, ids)
    qwen_layers   = load_features(qwen_path,   ids)

    per_monkey = {}
    with h5py.File(tvsd_path, "r") as f:
        for m in monkeys:
            roi_data = {r: f[f"test/neural_data/{m}/{r}"][:] for r in rois}
            res = compare_models_and_targets(
                {"ResNet": resnet_layers, "Qwen": qwen_layers},
                roi_data, rsa=rsa, cka=cka,
            )
            per_monkey[m] = scores_to_dataframe(res)
    return per_monkey, average_scores_df(per_monkey)


def nsd_alignment_multisubject(
    nsd_path: str,
    rois: list,
    resnet_path: str,
    qwen_path: str,
    subjects: list = None,
    rsa: RepresentationalSimilarityAnalysis = None,
    cka: CenteredKernelAlignment = None,
):
    """
    Run RSA/CKA alignment for each NSD subject. Each subject has its own
    stimulus subset. All subject IDs are collected first so features are
    loaded only once for the union, then indexed in-memory per subject.

    Returns
    -------
    per_subject : dict {subject: DataFrame}
    df_avg      : DataFrame averaged across subjects
    """
    if subjects is None:
        subjects = [f"subj0{i}" for i in range(1, 9)]

    subject_data = {}
    with h5py.File(nsd_path, "r") as f:
        for s in subjects:
            ids = f[f"test/stimulus_ids/{s}"][:]
            roi_data = {r: f[f"test/neural_data/{s}/{r}"][:] for r in rois}
            subject_data[s] = (roi_data, ids)

    all_ids = np.unique(np.concatenate([ids for _, ids in subject_data.values()]))
    resnet_layers = load_features(resnet_path, all_ids)
    qwen_layers   = load_features(qwen_path,   all_ids)
    all_id_to_idx = {id_: i for i, id_ in enumerate(all_ids)}

    per_subject = {}
    for s in subjects:
        roi_data, ids = subject_data[s]
        idx = np.array([all_id_to_idx[x] for x in ids])
        res = compare_models_and_targets(
            {"ResNet": {k: v[idx] for k, v in resnet_layers.items()},
             "Qwen":   {k: v[idx] for k, v in qwen_layers.items()}},
            roi_data, rsa=rsa, cka=cka,
        )
        per_subject[s] = scores_to_dataframe(res)
    return per_subject, average_scores_df(per_subject)


def eeg_timeresolved_multisubject(
    eeg_path: str,
    region: str,
    resnet_path: str,
    qwen_path: str,
    subjects: list = None,
    rsa: RepresentationalSimilarityAnalysis = None,
    cka: CenteredKernelAlignment = None,
):
    """
    Run EEG time-resolved alignment for each subject. Stimulus IDs are shared
    across EEG subjects, so model features are loaded once.

    Returns
    -------
    per_subject : dict {subject: {model: {'RSA': array, 'CKA': array}}}
    avg_scores  : {model: {'RSA': array, 'CKA': array}} averaged across subjects
    n_timepoints : int
    """
    if subjects is None:
        subjects = [f"sub-{i:02d}" for i in range(1, 11)]

    with h5py.File(eeg_path, "r") as f:
        ids = f["test/stimulus_ids"][:]
    resnet_layers = load_features(resnet_path, ids)
    qwen_layers   = load_features(qwen_path,   ids)

    per_subject  = {}
    n_timepoints = None
    with h5py.File(eeg_path, "r") as f:
        for s in subjects:
            eeg_data = f[f"test/neural_data/{s}/{region}"][:]
            n_timepoints = eeg_data.shape[2]
            per_subject[s] = compute_eeg_timeresolved(
                {"ResNet": resnet_layers, "Qwen": qwen_layers},
                eeg_data, rsa=rsa, cka=cka,
            )
    return per_subject, average_timeresolved(per_subject), n_timepoints


def plot_layer_roi_heatmap(
    df: pd.DataFrame,
    targets: list,
    title: str,
    metric: str = "RSA",
    save_path: str = None,
):
    """
    Heatmap of layer × ROI alignment scores for each model.

    Rows = layers in architectural order (early → late).
    Columns = ROIs ordered from low-level to high-level as given by `targets`.
    One subplot per model side by side.
    """
    models = sorted(df["model"].unique())
    ncols = len(models)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4.5))
    if ncols == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[(df["model"] == model) & (df["metric"] == metric)]
        layers = sort_layer_names(sub["layer"].unique().tolist())

        mat = np.full((len(layers), len(targets)), np.nan)
        for i, layer in enumerate(layers):
            for j, roi in enumerate(targets):
                row = sub[(sub["layer"] == layer) & (sub["target"] == roi)]
                if len(row) > 0:
                    mat[i, j] = row["score"].values[0]

        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0)
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(targets, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=7)
        ax.set_title(f"{title} — {model}, {metric}", fontsize=10, fontweight="bold")
        ax.set_xlabel("ROI (low → high level)")
        ax.set_ylabel("Layer (early → late)")
        plt.colorbar(im, ax=ax, label=metric)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_summary(
    dfs_by_dataset: dict,
    targets_by_dataset: dict,
    metrics: tuple = ("RSA", "CKA"),
    save_path: str = None,
):
    """
    Best-layer grouped bar chart comparing models across datasets and ROIs.

    Parameters
    ----------
    dfs_by_dataset : dict
        {dataset_label: DataFrame} from scores_to_dataframe.
    targets_by_dataset : dict
        {dataset_label: [roi, ...]} — ROI order per dataset.
    metrics : tuple
        Metrics to plot (one subplot each).
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    best : pd.DataFrame
        Best-layer score table (model, dataset, target, metric, layer, score).
    """
    frames = []
    for ds_label, df in dfs_by_dataset.items():
        frames.append(df.assign(dataset=ds_label))
    df_all = pd.concat(frames, ignore_index=True)

    idx  = df_all.groupby(["model", "dataset", "target", "metric"])["score"].idxmax()
    best = (
        df_all.loc[idx]
        .sort_values(["dataset", "target", "metric", "model"])
        .reset_index(drop=True)
    )

    print("=== Best-layer scores across datasets and targets ===")
    print(best[["model", "dataset", "target", "metric", "layer", "score"]].to_string(index=False))

    models  = sorted(df_all["model"].unique())
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors  = {m: palette[i] for i, m in enumerate(models)}

    datasets_order = [(ds, targets_by_dataset[ds]) for ds in dfs_by_dataset]

    ncols = min(len(metrics), 2)
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, metrics):
        sub = best[best["metric"] == metric]
        x_labels, x_centers, pos = [], [], 0
        for ds, targets in datasets_order:
            sub_ds = sub[sub["dataset"] == ds]
            for t in targets:
                sub_t = sub_ds[sub_ds["target"] == t]
                first = (ds == datasets_order[0][0] and t == datasets_order[0][1][0])
                for j, model in enumerate(models):
                    row   = sub_t[sub_t["model"] == model]
                    score = row["score"].values[0] if len(row) else 0
                    ax.bar(pos + j * 0.4, score, 0.35,
                           color=colors[model],
                           label=model if first else "")
                x_labels.append(f"{ds}\n{t}")
                x_centers.append(pos + 0.2)
                pos += 1.1
            pos += 0.5
        ax.set_xticks(x_centers)
        ax.set_xticklabels(x_labels, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("Best-layer score")
        ax.set_title(f"Model comparison — {metric}", fontsize=10, fontweight="bold")
        ax.legend()

    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return best, fig


def load_features(feat_path, neural_ids):
    with h5py.File(feat_path, "r") as f:
        feat_ids  = f["ids"][:]
        id_to_idx = {id_: i for i, id_ in enumerate(feat_ids)}
        feat_idx  = np.array([id_to_idx[x] for x in neural_ids])

        sort_order    = np.argsort(feat_idx)
        restore_order = np.argsort(sort_order)
        sorted_idx    = feat_idx[sort_order]

        layers = {}
        for key in f["features"].keys():
            data = f["features"][key][sorted_idx]
            layers[key] = data[restore_order]
    return layers


