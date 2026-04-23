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