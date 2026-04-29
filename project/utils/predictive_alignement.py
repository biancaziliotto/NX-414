import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import h5py

class ModelBrainDataset():
    """
    Dataset class for handling model-based encoding tasks.
    Loads model activations for given stimuli and pairs them with neural responses.
    Assumes train and test sets are already separated.
    """
    def __init__(self, y_train, y_test, stimuli_train, stimuli_test, model_name, dataset_name, layer_name):
        """
        Initializes the dataset by loading model activations for provided stimuli.

        Parameters:
        y_train (array-like): Training neural response values (n_train_samples, n_units).
        y_test (array-like): Test neural response values (n_test_samples, n_units).
        stimuli_train (array-like): Stimuli identifiers/indices for training (n_train_samples,).
        stimuli_test (array-like): Stimuli identifiers/indices for test (n_test_samples,).
        model_name (str): Name of the model (used to locate activations).
        activations_path (str): Path for loading activations.
        layer_name (str): Name of layer.
        """
        self.y_train = y_train
        self.y_test = y_test
        self.stimuli_train = stimuli_train
        self.stimuli_test = stimuli_test
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.activations_path = f"/shared/NX-414/extracted_features/{model_name}/{dataset_name}.h5"
        
        # Load activations
        self.X_train = self._load_activations(stimuli_train, layer_name)
        self.X_test = self._load_activations(stimuli_test, layer_name)
        
        self.X_val = None
        self.y_val = None
        self.X_train_split = None
        self.y_train_split = None

    def _load_activations(self, stimuli_ids, layer):
        """
        Loads model activations for the given stimuli.

        Parameters:
        stimuli_ids (array-like): Stimuli identifiers/indices.
        model_name (str): Name of the model.
        path_template (str): Template path for loading activations.

        Returns:
        array-like: Stacked activations (n_samples, n_features).
        """
         
        activations_file = h5py.File(self.activations_path, "r")
        
        feat_ids = list(activations_file["ids"])
        id_to_feat_idx = {id_: i for i, id_ in enumerate(feat_ids)}
        feat_idx = np.array([id_to_feat_idx[x] for x in stimuli_ids])

        layer_act = activations_file["features"][layer]
        
        # HDF5 requires indices in sorted order for fancy indexing
        # So we sort, read, then un-sort to match original order
        sort_order = np.argsort(feat_idx)
        feat_idx_sorted = feat_idx[sort_order]
        
        # Read with sorted indices (vectorized, fast!)
        X_sorted = layer_act[feat_idx_sorted, :]
        
        # Un-sort to match original stimulus order
        unsort_order = np.argsort(sort_order)
        X = X_sorted[unsort_order, :]

        return X

    def get_data(self):
        """
        Returns the training and test data.

        Returns:
        tuple: (X_train, y_train, X_test, y_test)
        """
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def split_train_val(self, val_size=0.2, random_state=42):
        """
        Splits the training data into train and validation sets.

        Parameters:
        val_size (float): Proportion of training data for validation set.
        random_state (int): Random seed for reproducibility.
        """
        self.X_train_split, self.X_val, self.y_train_split, self.y_val = train_test_split(
            self.X_train, self.y_train,
            test_size=val_size, random_state=random_state
        )
    
    def get_train_val_splits(self):
        """
        Returns the train/val splits (after calling split_train_val).

        Returns:
        dict: Dictionary with 'X_train', 'X_val', 'y_train', 'y_val'.
        """
        return {
            'X_train': self.X_train_split,
            'X_val': self.X_val,
            'y_train': self.y_train_split,
            'y_val': self.y_val
        }
    
    def __len__(self):
        """
        Returns the number of training samples in the dataset.

        Returns:
        int: Number of training samples.
        """
        return len(self.X_train)


class LinearRegressionModel(nn.Module):
    """Simple linear regression model for GPU acceleration."""
    def __init__(self, n_features, n_outputs):
        super().__init__()
        self.linear = nn.Linear(n_features, n_outputs)
    
    def forward(self, x):
        return self.linear(x)


class SGDEncoder():
    """
    Linear Encoding Model using PyTorch with GPU acceleration.
    Supports multi-output regression for multiple neural units.
    """
    def __init__(self, alpha=0.0001, max_iter=500, min_iter=10, batch_size=32, learning_rate=0.001, 
                 early_stopping_patience=10, early_stopping_tol=1e-6, random_state=42):
        """
        Initializes the SGDEncoder with GPU support.

        Parameters:
        alpha (float): L2 regularization strength (default: 0.0001).
        max_iter (int): Maximum number of iterations/epochs (default: 500).
        min_iter (int): Minimum number of epochs before early stopping allowed (default: 10).
        batch_size (int): Batch size for training (default: 32).
        learning_rate (float): Learning rate for the optimizer (default: 0.001 for stability).
        early_stopping_patience (int): Number of epochs with no improvement to stop training (default: 10).
        early_stopping_tol (float): Tolerance for improvement detection (default: 1e-6).
        random_state (int): Random seed for reproducibility.
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_tol = early_stopping_tol
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.best_alpha_ = None
        self.cv_results_ = None
        self.X_scaler = None
        self.y_scaler = None
        
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ GPU not available, using CPU")

    def _create_model(self, n_features, n_outputs):
        """Create a fresh model instance."""
        model = LinearRegressionModel(n_features, n_outputs).to(self.device)
        return model

    def fit(self, X, y, batch_size=None, verbose=False):
        """
        Fits the model to the training data on GPU with data normalization and early stopping.

        Parameters:
        X (array-like): Feature vectors (n_samples, n_features).
        y (array-like): Target values (n_samples,) or (n_samples, n_units) for multi-output.
        batch_size (int): Batch size for training. If None, uses self.batch_size.
        verbose (bool): Print training progress.
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Normalize data for numerical stability
        if verbose:
            print("  Normalizing data...")
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        X_normalized = self.X_scaler.fit_transform(X)
        y_normalized = self.y_scaler.fit_transform(y)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_normalized).to(self.device)
        y_tensor = torch.FloatTensor(y_normalized).to(self.device)
        
        if len(y_tensor.shape) == 1:
            y_tensor = y_tensor.unsqueeze(1)
        
        # Create model
        self.model = self._create_model(X_normalized.shape[1], y_tensor.shape[1])
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Early stopping tracking
        best_loss = float('inf')
        patience_counter = 0
        
        # Training loop with early stopping
        for epoch in range(self.max_iter):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                y_pred = self.model(batch_X)
                loss = criterion(y_pred, batch_y)
                
                # Add L2 regularization
                l2_reg = 0
                for param in self.model.parameters():
                    l2_reg += torch.sum(param ** 2)
                loss += self.alpha * l2_reg
                
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            # Early stopping check (only after min_iter epochs)
            if epoch >= self.min_iter - 1:
                if avg_loss < best_loss - self.early_stopping_tol:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                best_loss = min(best_loss, avg_loss)
            
            if verbose and (epoch + 1) % max(1, self.max_iter // 50) == 0:
                print(f"  Epoch {epoch + 1}/{self.max_iter}, Loss: {avg_loss:.6f}")
            
            # Stop if converged (only after min_iter epochs)
            if epoch >= self.min_iter - 1 and patience_counter >= self.early_stopping_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1} (no improvement for {self.early_stopping_patience} epochs, tol={self.early_stopping_tol:.1e})")
                break

    def predict(self, X):
        """
        Predicts target values using the fitted model.
        Applies inverse normalization to return predictions in original scale.

        Parameters:
        X (array-like): Feature vectors (n_samples, n_features).

        Returns:
        array-like: Predicted values (n_samples,) or (n_samples, n_units) for multi-output.
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        self.model.eval()
        # Normalize input
        X_normalized = self.X_scaler.transform(X)
        X_tensor = torch.FloatTensor(X_normalized).to(self.device)
        
        with torch.no_grad():
            y_pred_normalized = self.model(X_tensor).cpu().numpy()
        
        # Inverse normalize predictions
        y_pred = self.y_scaler.inverse_transform(y_pred_normalized)
        
        return y_pred
    
    def cross_validate(self, X, y, cv=5, scoring='r2', verbose=False):
        """
        Performs cross-validation on the model.

        Parameters:
        X (array-like): Feature vectors.
        y (array-like): Target values (n_samples,) or (n_samples, n_units).
        cv (int): Number of cross-validation folds.
        scoring (str): Scoring metric (default: 'r2').
        verbose (bool): Print progress.

        Returns:
        dict: Cross-validation scores.
        """
        scores = []
        n_samples = len(X)
        fold_size = n_samples // cv
        
        for fold in range(cv):
            # Create fold indices
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < cv - 1 else n_samples
            
            X_train = np.vstack([X[:test_start], X[test_end:]])
            X_val = X[test_start:test_end]
            y_train = np.vstack([y[:test_start], y[test_end:]]) if len(y.shape) > 1 else np.concatenate([y[:test_start], y[test_end:]])
            y_val = y[test_start:test_end]
            
            # Train and score
            self.fit(X_train, y_train, verbose=verbose)
            y_pred = self.predict(X_val)
            
            if len(y_val.shape) == 1:
                fold_score = r2_score(y_val, y_pred)
            else:
                fold_score = r2_score(y_val, y_pred)
            
            scores.append(fold_score)
        
        self.cv_results_ = {
            'scores': np.array(scores),
            'mean': np.mean(scores),
            'std': np.std(scores)
        }
        return self.cv_results_
    
    def select_hyperparams(self, X, y, alphas=None, cv=5, scoring='r2', verbose=False):
        """
        Performs hyperparameter selection using cross-validation.

        Parameters:
        X (array-like): Feature vectors.
        y (array-like): Target values (n_samples,) or (n_samples, n_units).
        alphas (array-like): Alpha values to search. Default: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1].
        cv (int): Number of cross-validation folds.
        scoring (str): Scoring metric (default: 'r2').
        verbose (bool): Print progress.

        Returns:
        dict: Best parameters and corresponding score.
        """
        if alphas is None:
            # Stronger regularization to prevent overfitting with high-dim features
            alphas = [1e-3, 1e-2, 1e-1, 1, 10]
        
        best_score = -np.inf
        best_alpha = None
        all_scores = {}
        
        for alpha in alphas:
            self.alpha = alpha
            cv_results = self.cross_validate(X, y, cv=cv, scoring=scoring, verbose=verbose)
            mean_score = cv_results['mean']
            all_scores[alpha] = mean_score
            
            if verbose:
                print(f"  Alpha {alpha:.1e}: CV R² = {mean_score:.4f} ± {cv_results['std']:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha
        
        self.best_alpha_ = best_alpha
        self.alpha = best_alpha
        
        return {
            'best_alpha': best_alpha,
            'best_score': best_score,
            'cv_results': all_scores
        }
    
    def select_hyperparams_simple(self, X, y, val_size=0.2, alphas=None, scoring='r2', verbose=False):
        """
        Performs hyperparameter selection using a simple train-validation split (no cross-validation).
        This is faster than cross-validation but uses less data for training.
        
        Parameters:
        X (array-like): Feature vectors.
        y (array-like): Target values (n_samples,) or (n_samples, n_units).
        val_size (float): Proportion of data to use for validation.
        alphas (array-like): Alpha values to search. Default: [1e-3, 1e-2, 1e-1, 1, 10].
        scoring (str): Scoring metric (default: 'r2').
        verbose (bool): Print progress.
        
        Returns:
        dict: Best parameters and corresponding score.
        """
        if alphas is None:
            alphas = [1e-3, 1e-2, 1e-1, 1, 10]
        
        # Simple train-val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=self.random_state
        )
        
        best_score = -np.inf
        best_alpha = None
        all_scores = {}
        
        for alpha in alphas:
            self.alpha = alpha
            # Train on split
            self.fit(X_train, y_train, verbose=False)
            # Evaluate on validation
            y_pred = self.predict(X_val)
            
            if scoring == 'r2':
                score = r2_score(y_val, y_pred)
            elif scoring == 'mse':
                score = -mean_squared_error(y_val, y_pred)  # negative for consistency
            else:
                score = r2_score(y_val, y_pred)
            
            all_scores[alpha] = score
            
            if verbose:
                print(f"  Alpha {alpha:.1e}: Val R² = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_alpha = alpha
        
        self.best_alpha_ = best_alpha
        self.alpha = best_alpha
        
        return {
            'best_alpha': best_alpha,
            'best_score': best_score,
            'cv_results': all_scores
        }
    
    def fit_and_evaluate(self, dataset, alphas=None, cv=5, val_size=0.2, scoring='r2', verbose=False, use_cv=True):
        """
        Fit and evaluate model on a ModelBrainDataset with GPU acceleration.
        
        Workflow:
        1. Split training data into train+val
        2. Select hyperparameters using cross-validation or simple train-val split
        3. Refit on full train+val
        4. Evaluate on held-out test set

        Parameters:
        dataset (ModelBrainDataset): Dataset with pre-separated train and test splits.
        alphas (array-like): Alpha values to search. If None, uses defaults.
        cv (int): Number of cross-validation folds (used only if use_cv=True).
        val_size (float): Proportion of training data for validation.
        scoring (str): Scoring metric (default: 'r2').
        verbose (bool): Print progress.
        use_cv (bool): If True, uses cross-validation for hyperparameter selection.
                       If False, uses simple train-val split (faster). Default: True.

        Returns:
        dict: Results including best_alpha, cv_score, r2_test, mse_test, predictions.
        """
        # Split training data into train and validation
        dataset.split_train_val(val_size=val_size, random_state=self.random_state)
        splits = dataset.get_train_val_splits()
        
        # Combine train and val for hyperparameter selection
        X_train_val = np.vstack([splits['X_train'], splits['X_val']])
        y_train_val = np.vstack([splits['y_train'], splits['y_val']])
        
        # Get test data (completely held-out)
        X_test = dataset.X_test
        y_test = dataset.y_test
        
        # Select hyperparameters using chosen method
        if verbose:
            if use_cv:
                print(f"Selecting hyperparameters with {cv}-fold cross-validation...")
            else:
                print("Selecting hyperparameters with simple train-val split...")
        
        if use_cv:
            hp_results = self.select_hyperparams(
                X_train_val, y_train_val,
                alphas=alphas, cv=cv, scoring=scoring, verbose=verbose
            )
        else:
            hp_results = self.select_hyperparams_simple(
                X_train_val, y_train_val,
                val_size=val_size, alphas=alphas, scoring=scoring, verbose=verbose
            )
        
        # Refit on full train+val
        if verbose:
            print(f"Refitting with best alpha={hp_results['best_alpha']:.1e}...")
        self.fit(X_train_val, y_train_val, verbose=verbose)
        
        # Evaluate on test set
        if verbose:
            print("Evaluating on test set...")
        y_pred_test = self.predict(X_test)
        r2_test = r2_score(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        
        return {
            'best_alpha': hp_results['best_alpha'],
            'hp_score': hp_results['best_score'],
            'r2_test': r2_test,
            'mse_test': mse_test,
            'y_pred_test': y_pred_test,
            'y_test': y_test
        }
