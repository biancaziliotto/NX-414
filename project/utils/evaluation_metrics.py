"""
Evaluation metrics for neural encoding models.
Includes standard metrics (R², MSE) and advanced metrics (Pearson correlation, 
explained variance, and their noise-corrected variants).
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score


def compute_pearson_correlation(y_true, y_pred):
    """
    Compute Pearson correlation between predictions and ground truth (per unit).
    
    Parameters:
    y_true (array-like): Ground truth values (n_samples, n_units)
    y_pred (array-like): Predictions (n_samples, n_units)
    
    Returns:
    tuple: (correlations, pvalues) - arrays of correlation coefficients and p-values
    """
    correlations = []
    pvalues = []
    
    for i in range(y_true.shape[1]):
        corr, pval = pearsonr(y_true[:, i], y_pred[:, i])
        correlations.append(corr)
        pvalues.append(pval)
    
    return np.array(correlations), np.array(pvalues)


def compute_explained_variance(y_true, y_pred):
    """
    Compute explained variance score per unit.
    
    Parameters:
    y_true (array-like): Ground truth values (n_samples, n_units)
    y_pred (array-like): Predictions (n_samples, n_units)
    
    Returns:
    array-like: Explained variance per unit
    """
    explained_var = []
    
    for i in range(y_true.shape[1]):
        ev = explained_variance_score(y_true[:, i], y_pred[:, i])
        explained_var.append(ev)
    
    return np.array(explained_var)


def compute_noise_ceiling(y_true, ceiling_value=0.85):
    """
    Estimate noise ceiling from measurement reliability.
    For neural data, assumes split-half reliability or test-retest correlation.
    
    Parameters:
    y_true (array-like): Ground truth values (n_samples, n_units)
    ceiling_value (float): Assumed noise ceiling value for neural measurements (default: 0.85)
    
    Returns:
    array-like: Noise ceiling per unit
    """
    n_units = y_true.shape[1]
    # Conservative noise ceiling estimate for neural data
    # This should be calibrated with actual split-half correlations if available
    noise_ceiling = np.ones(n_units) * ceiling_value
    
    return noise_ceiling


def compute_noise_corrected_pearson(y_true, y_pred, noise_ceiling=None):
    """
    Compute noise-corrected Pearson correlation.
    Correction: corr_corrected = corr / sqrt(noise_ceiling)
    
    Parameters:
    y_true (array-like): Ground truth values (n_samples, n_units)
    y_pred (array-like): Predictions (n_samples, n_units)
    noise_ceiling (array-like): Per-unit noise ceiling. If None, estimates from data.
    
    Returns:
    array-like: Noise-corrected correlations (clipped to [-1, 1])
    """
    if noise_ceiling is None:
        noise_ceiling = compute_noise_ceiling(y_true)
    
    correlations, _ = compute_pearson_correlation(y_true, y_pred)
    noise_ceiling = np.clip(noise_ceiling, 0.01, 0.99)
    
    noise_corrected = correlations / np.sqrt(noise_ceiling)
    noise_corrected = np.clip(noise_corrected, -1.0, 1.0)
    
    return noise_corrected


def compute_noise_corrected_explained_variance(y_true, y_pred, noise_ceiling=None):
    """
    Compute noise-corrected explained variance.
    Correction: ev_corrected = ev / noise_ceiling
    
    Parameters:
    y_true (array-like): Ground truth values (n_samples, n_units)
    y_pred (array-like): Predictions (n_samples, n_units)
    noise_ceiling (array-like): Per-unit noise ceiling. If None, estimates from data.
    
    Returns:
    array-like: Noise-corrected explained variance (clipped to [0, 1])
    """
    if noise_ceiling is None:
        noise_ceiling = compute_noise_ceiling(y_true)
    
    explained_var = compute_explained_variance(y_true, y_pred)
    noise_ceiling = np.clip(noise_ceiling, 0.01, 0.99)
    
    noise_corrected_ev = explained_var / noise_ceiling
    noise_corrected_ev = np.clip(noise_corrected_ev, 0.0, 1.0)
    
    return noise_corrected_ev


def compute_all_metrics(y_true, y_pred, noise_ceiling=None):
    """
    Compute all evaluation metrics at once.
    
    Parameters:
    y_true (array-like): Ground truth values (n_samples, n_units)
    y_pred (array-like): Predictions (n_samples, n_units)
    noise_ceiling (array-like): Per-unit noise ceiling. If None, estimates from data.
    
    Returns:
    dict: Dictionary with all computed metrics (mean, std, median, min, max for each metric type)
    """
    if noise_ceiling is None:
        noise_ceiling = compute_noise_ceiling(y_true)
    
    # Compute all metrics
    pearson_corr = compute_pearson_correlation(y_true, y_pred)[0]
    explained_var = compute_explained_variance(y_true, y_pred)
    noise_corr_pearson = compute_noise_corrected_pearson(y_true, y_pred, noise_ceiling)
    noise_corr_ev = compute_noise_corrected_explained_variance(y_true, y_pred, noise_ceiling)
    
    return {
        'pearson_corr_mean': np.mean(pearson_corr),
        'pearson_corr_std': np.std(pearson_corr),
        'pearson_corr_median': np.median(pearson_corr),
        'pearson_corr_min': np.min(pearson_corr),
        'pearson_corr_max': np.max(pearson_corr),
        'explained_var_mean': np.mean(explained_var),
        'explained_var_std': np.std(explained_var),
        'explained_var_median': np.median(explained_var),
        'explained_var_min': np.min(explained_var),
        'explained_var_max': np.max(explained_var),
        'noise_corrected_pearson_mean': np.mean(noise_corr_pearson),
        'noise_corrected_pearson_std': np.std(noise_corr_pearson),
        'noise_corrected_pearson_median': np.median(noise_corr_pearson),
        'noise_corrected_pearson_min': np.min(noise_corr_pearson),
        'noise_corrected_pearson_max': np.max(noise_corr_pearson),
        'noise_corrected_ev_mean': np.mean(noise_corr_ev),
        'noise_corrected_ev_std': np.std(noise_corr_ev),
        'noise_corrected_ev_median': np.median(noise_corr_ev),
        'noise_corrected_ev_min': np.min(noise_corr_ev),
        'noise_corrected_ev_max': np.max(noise_corr_ev),
        'noise_ceiling_mean': np.mean(noise_ceiling),
    }
