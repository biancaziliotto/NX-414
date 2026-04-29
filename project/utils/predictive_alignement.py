import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
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
        Loads model activations for the given stimuli (parallel).

        Parameters:
        stimuli_ids (array-like): Stimuli identifiers/indices.
        model_name (str): Name of the model.
        path_template (str): Template path for loading activations.

        Returns:
        array-like: Stacked activations (n_samples, n_features).
        """
         
        activations_file = h5py.File(self.activations_path, "r")
        layers = list(activations_file["features"].keys())
        
        feat_ids = list(activations_file["ids"])
        id_to_feat_idx = {id_: i for i, id_ in enumerate(feat_ids)}
        feat_idx = np.array([id_to_feat_idx[x] for x in stimuli_ids])

        layer_act = activations_file["features"][layer]
        activations_list = [layer_act[feat_id, :] for feat_id in feat_idx]
        
        # Stack all activations
        X = np.vstack(activations_list)
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


def _train_and_score_fold(X_train, X_val, y_train, y_val, alpha, learning_rate, batch_size, 
                           max_iter, min_iter, early_stopping_patience, early_stopping_tol, 
                           random_state, verbose=False):
    """
    Helper function to train and evaluate a single fold.
    Runs on CPU to avoid GPU memory conflicts during parallel CV.
    """
    device = torch.device('cpu')  # Use CPU for parallel CV
    
    # Normalize data
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_norm = X_scaler.fit_transform(X_train)
    y_train_norm = y_scaler.fit_transform(y_train)
    X_val_norm = X_scaler.transform(X_val)
    y_val_norm = y_scaler.transform(y_val)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_norm).to(device)
    y_train_tensor = torch.FloatTensor(y_train_norm).to(device)
    if len(y_train_tensor.shape) == 1:
        y_train_tensor = y_train_tensor.unsqueeze(1)
    
    X_val_tensor = torch.FloatTensor(X_val_norm).to(device)
    y_val_tensor = torch.FloatTensor(y_val_norm).to(device)
    if len(y_val_tensor.shape) == 1:
        y_val_tensor = y_val_tensor.unsqueeze(1)
    
    # Create model
    model = LinearRegressionModel(X_train_norm.shape[1], y_train_tensor.shape[1]).to(device)
    
    # Create dataloader
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer and loss
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(max_iter):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            
            # L2 regularization
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.sum(param ** 2)
            loss += alpha * l2_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        # Early stopping
        if epoch >= min_iter - 1:
            if avg_loss < best_loss - early_stopping_tol:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            best_loss = min(best_loss, avg_loss)
        
        if patience_counter >= early_stopping_patience and epoch >= min_iter - 1:
            break
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        y_val_pred_norm = model(X_val_tensor).cpu().numpy()
    
    # Inverse transform back to original scale
    y_val_pred = y_scaler.inverse_transform(y_val_pred_norm)
    
    # Calculate R² score
    fold_score = r2_score(y_val, y_val_pred)
    
    return fold_score


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
    
    def cross_validate(self, X, y, cv=5, scoring='r2', verbose=False, n_jobs=-1):
        """
        Performs cross-validation on the model (parallelized across folds).

        Parameters:
        X (array-like): Feature vectors.
        y (array-like): Target values (n_samples,) or (n_samples, n_units).
        cv (int): Number of cross-validation folds.
        scoring (str): Scoring metric (default: 'r2').
        verbose (bool): Print progress.
        n_jobs (int): Number of parallel jobs (-1 for all CPUs).

        Returns:
        dict: Cross-validation scores.
        """
        n_samples = len(X)
        fold_size = n_samples // cv
        
        # Prepare fold data
        folds = []
        for fold in range(cv):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < cv - 1 else n_samples
            
            X_train = np.vstack([X[:test_start], X[test_end:]])
            X_val = X[test_start:test_end]
            y_train = np.vstack([y[:test_start], y[test_end:]]) if len(y.shape) > 1 else np.concatenate([y[:test_start], y[test_end:]])
            y_val = y[test_start:test_end]
            
            folds.append((X_train, X_val, y_train, y_val))
        
        # Parallel CV training
        scores = Parallel(n_jobs=n_jobs, backend='threading')(delayed(_train_and_score_fold)(
            X_train, X_val, y_train, y_val,
            self.alpha, self.learning_rate, self.batch_size,
            self.max_iter, self.min_iter, self.early_stopping_patience, self.early_stopping_tol,
            self.random_state, verbose=verbose
        ) for X_train, X_val, y_train, y_val in folds)
        
        self.cv_results_ = {
            'scores': np.array(scores),
            'mean': np.mean(scores),
            'std': np.std(scores)
        }
        return self.cv_results_
    
    def select_hyperparams(self, X, y, alphas=None, cv=5, scoring='r2', verbose=False, n_jobs=-1):
        """
        Performs hyperparameter selection using cross-validation (parallelized across alphas).

        Parameters:
        X (array-like): Feature vectors.
        y (array-like): Target values (n_samples,) or (n_samples, n_units).
        alphas (array-like): Alpha values to search. Default: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1].
        cv (int): Number of cross-validation folds.
        scoring (str): Scoring metric (default: 'r2').
        verbose (bool): Print progress.
        n_jobs (int): Number of parallel jobs for alpha search (-1 for all CPUs).

        Returns:
        dict: Best parameters and corresponding score.
        """
        if alphas is None:
            # Stronger regularization to prevent overfitting with high-dim features
            alphas = [1e-3, 1e-2, 1e-1, 1, 10]
        
        # Helper function for parallel alpha evaluation
        def evaluate_alpha(alpha):
            temp_encoder = SGDEncoder(
                alpha=alpha,
                max_iter=self.max_iter,
                min_iter=self.min_iter,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                early_stopping_patience=self.early_stopping_patience,
                early_stopping_tol=self.early_stopping_tol,
                random_state=self.random_state
            )
            cv_results = temp_encoder.cross_validate(X, y, cv=cv, scoring=scoring, verbose=False, n_jobs=1)
            return alpha, cv_results['mean'], cv_results['std'], cv_results
        
        # Parallel alpha search
        results = Parallel(n_jobs=n_jobs, backend='threading')(delayed(evaluate_alpha)(alpha) for alpha in alphas)
        
        best_score = -np.inf
        best_alpha = None
        all_scores = {}
        
        for alpha, mean_score, std_score, cv_results in results:
            all_scores[alpha] = mean_score
            
            if verbose:
                print(f"  Alpha {alpha:.1e}: CV R² = {mean_score:.4f} ± {std_score:.4f}")
            
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
    
    def fit_and_evaluate(self, dataset, alphas=None, cv=5, val_size=0.2, scoring='r2', verbose=False, n_jobs=-1):
        """
        Fit and evaluate model on a ModelBrainDataset with GPU acceleration.
        
        Workflow:
        1. Split training data into train+val
        2. Select hyperparameters using cross-validation
        3. Refit on full train+val
        4. Evaluate on held-out test set

        Parameters:
        dataset (ModelBrainDataset): Dataset with pre-separated train and test splits.
        alphas (array-like): Alpha values to search. If None, uses defaults.
        cv (int): Number of cross-validation folds for hyperparameter selection.
        val_size (float): Proportion of training data for validation.
        scoring (str): Scoring metric (default: 'r2').
        verbose (bool): Print progress.

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
        
        # Select hyperparameters on train+val (no test leakage)
        if verbose:
            print("Selecting hyperparameters...")
        hp_results = self.select_hyperparams(
            X_train_val, y_train_val,
            alphas=alphas, cv=cv, scoring=scoring, verbose=verbose, n_jobs=n_jobs
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
            'cv_score': hp_results['best_score'],
            'r2_test': r2_test,
            'mse_test': mse_test,
            'y_pred_test': y_pred_test,
            'y_test': y_test
        }
