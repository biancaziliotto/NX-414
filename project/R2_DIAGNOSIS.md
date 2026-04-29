# Negative R² Diagnosis - Root Causes & Solutions

## Problems Identified:

### 1. **Insufficient Regularization (MAIN ISSUE)**
- Default alpha range [1e-6, 1e-5, 1e-4, 1e-3, 1e-2] is TOO WEAK
- With 30,000 features and only ~100 test samples, model BADLY overfits
- Weak regularization means no penalty for learning noise
- **Solution**: Use stronger alphas [1e-3, 1e-2, 1e-1, 1, 10]

### 2. **Early Stopping Stops Training Too Soon**
- `min_epochs=20` with `patience=10` means stops after epoch 30 at minimum
- Model hasn't learned meaningful patterns yet
- Loss decreases smoothly but R² stays negative (learning noise, not signal)
- **Solution**: 
  - Increase `min_epochs` to 50-100
  - Decrease `patience` from 10 to 5 (but only after min_epochs)
  - Increase `max_epochs` to 1000

### 3. **Learning Rate May Be Wrong**
- 0.001 is good but with strong regularization, might need tuning
- Could use learning rate scheduler (decrease over time)

### 4. **Small Test Set (100 samples)**
- Only 100 holdout examples for 241+ neurons
- High variance in per-neuron R² values
- Some neurons naturally harder to predict

### 5. **No Validation During Training**
- No early stopping based on validation R², only training loss
- Model stops when training loss plateaus, but that doesn't guarantee good test R²

---

## Recommended Fixes:

### Fix 1: Better Alpha Range
```python
alphas = [1e-3, 1e-2, 1e-1, 1, 10]  # STRONG regularization
```

### Fix 2: Better Early Stopping
```python
encoder = SGDEncoder(
    alpha=0.0001,
    max_epochs=1000,      # More exploration
    min_epochs=100,       # Let it train longer before stopping
    patience=15,          # More patient with new direction
    early_stopping_tol=1e-4,  # Less strict convergence
    batch_size=256,
    learning_rate=0.001,
    random_state=42
)
```

### Fix 3: Diagnostic Reporting
Add reporting for:
- How many neurons have negative R²
- Which alphas give best CV scores
- Warning when test R² < CV R² (overfitting indicator)

### Fix 4: Data Insights
Check:
- Is the data even predictable? (check noise ceiling)
- Are there any units that are always 0 or have zero variance?
- Is train/test split representative?

---

## How to Run with Better Settings:

```bash
python train_encoding_models.py \
  --model Qwen3-VL-2B-Instruct \
  --dataset TVSD \
  --roi IT \
  --max-epochs 1000 \
  --min-epochs 100 \
  --patience 15 \
  --learning-rate 0.0005 \
  --verbose
```

---

## Expected Behavior After Fixes:
- ✓ Positive R² for most neurons
- ✓ CV R² and Test R² closer together
- ✓ Training converges around epoch 200-500 (not 20-30)
- ✓ Better alpha chosen (0.01 to 0.1 range likely)
