# RSA/CKA in Result JSON Files

## Overview

The project now supports saving both **feature-space** and **encoding-space** representational metrics (RSA, CKA) in the same JSON result files alongside predictive metrics.

## Metrics Available

### Already in Results
- `r2_mean`, `r2_std`, `r2_min`, `r2_max`, `r2_median` – R² scores
- `n_units`, `mse_mean`, `mse_std` – Basic evaluation results

### To Be Augmented via `augment_results.py`

**Predictive metrics (per-unit, summarized):**
- `pearson_corr_mean`, `pearson_corr_std`, `pearson_corr_min`, `pearson_corr_max`, `pearson_corr_median`
- `explained_var_mean`, `explained_var_std`, `explained_var_min`, `explained_var_max`, `explained_var_median`
- `noise_corrected_pearson_mean`, `noise_corrected_pearson_std`, etc.
- `noise_corrected_ev_mean`, `noise_corrected_ev_std`, etc.
- `noise_ceiling_mean`

**Representational metrics:**
- `feature_rsa` – RSA between model features and neural responses
- `feature_cka` – CKA between model features and neural responses
- `encoding_rsa` – RSA between predicted and actual neural responses
- `encoding_cka` – CKA between predicted and actual neural responses

## Usage

### Augment Existing Result JSON Files

To add all missing metrics (predictive + representational) to your result files:

```bash
python augment_results.py --results-dir ./results --base-dir .
```

This will:
1. Load each result entry from `results/*.json`
2. Re-run inference using saved model weights
3. Compute all missing metrics
4. Save updated results back to the JSON files (idempotent - entries with all keys are skipped)

### Generate Ranking Comparison Plots

Once augmented, you can create ranking comparison visualizations:

```bash
python plot_results.py
```

This generates figures like:
- `2_5_{dataset}_ranking_comparison_{roi}.png` – Compare metric rankings per ROI
- `2_5_{dataset}_ranking_agreement.png` – Best-layer agreement matrix per metric

### Use Metrics in Analysis

In the notebook, load and use these metrics:

```python
from utils.predictive_plots import load_predictive_results

# Load all results with their metrics
df = load_predictive_results(results_dir="results")

# Filter by metric type
feature_rsa_scores = df[df["metric"] == "feature_rsa"]
encoding_rsa_scores = df[df["metric"] == "encoding_rsa"]

# Compare rankings across metrics
metrics_to_compare = [
    "pearson_corr_mean",
    "explained_var_mean",
    "feature_rsa",
    "feature_cka",
    "encoding_rsa",
    "encoding_cka"
]
```

## Implementation Details

- **RSA and CKA classes**: Uses existing `RepresentationalSimilarityAnalysis` and `CenteredKernelAlignment` from `utils/alignement_utils.py`
- **Feature space**: Compares model layer features directly to neural responses
- **Encoding space**: Compares predicted responses (from linear encoding model) to actual neural responses
- **Idempotent**: `augment_results.py` only computes metrics that are missing, allowing safe re-runs

## Key Differences

| Metric | Computes | Interpretation |
|--------|----------|-----------------|
| `feature_rsa/cka` | RSA/CKA(model features, neural responses) | How well do model features capture neural representational structure? |
| `encoding_rsa/cka` | RSA/CKA(predicted responses, actual responses) | How well does the fitted linear encoding model preserve neural representational geometry? |
| `pearson_corr` | Per-unit correlation (predicted vs actual) | How well do individual units/voxels match predictions? |
| `encoding_rsa` vs `feature_rsa` | Indirectly comparable | Encoding adds a linear mapping step; feature compares raw features |

