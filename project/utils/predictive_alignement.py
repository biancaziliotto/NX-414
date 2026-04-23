import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import h5py

class ModelBrainDataset():
    """
    Dataset class for handling model-based encoding tasks.
    Loads model activations for given stimuli and pairs them with neural responses.
    Assumes train and test sets are already separated.
    """
    def __init__(self, y_train, y_test, stimuli_train, stimuli_test, model_name, dataset_name):
        """
        Initializes the dataset by loading model activations for provided stimuli.

        Parameters:
        y_train (array-like): Training neural response values (n_train_samples, n_units).
        y_test (array-like): Test neural response values (n_test_samples, n_units).
        stimuli_train (array-like): Stimuli identifiers/indices for training (n_train_samples,).
        stimuli_test (array-like): Stimuli identifiers/indices for test (n_test_samples,).
        model_name (str): Name of the model (used to locate activations).
        activations_path (str): Path for loading activations."
        """
        self.y_train = y_train
        self.y_test = y_test
        self.stimuli_train = stimuli_train
        self.stimuli_test = stimuli_test
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.activations_path = f"/shared/NX-414/extracted_features/{model_name}/{dataset_name}.h5"
        
        # Load activations
        self.X_train = self._load_activations(stimuli_train)
        self.X_test = self._load_activations(stimuli_test)
        
        self.X_val = None
        self.y_val = None
        self.X_train_split = None
        self.y_train_split = None

    def _load_activations(self, stimuli_ids):
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
        
        activation_indexes = list(activations_file["ids"])
        data_indexes = [activation_indexes.index(stimulus_id) for stimulus_id in stimuli_ids]      

        layer_act = activations_file["features"][layers[0]]
        print(len(activation_indexes))
        activations_list = [layer_act[data_idx, :] for data_idx in data_indexes]
        
        # Stack all activations
        X = np.vstack(activations_list)
        print(X.shape)
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

class SGDEncoder():
    """
    Linear Encoding Model using Stochastic Gradient Descent (SGD).
    """
    def __init__(self, alpha=0.0001, max_iter=1000, tol=1e-3, random_state=42):
        """
        Initializes the SGDEncoder.

        Parameters:
        alpha (float): L2 regularization strength (default: 0.0001).
        max_iter (int): Maximum number of iterations.
        tol (float): Stopping criterion tolerance.
        random_state (int): Random seed for reproducibility.
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.model = SGDRegressor(
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        self.cv_results_ = None
        self.best_alpha_ = None

    def fit(self, X, y):
        """
        Fits the model to the training data.

        Parameters:
        X (array-like): Feature vectors (n_samples, n_features).
        y (array-like): Target values (n_samples, n_uni).
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predicts target values using the fitted model.

        Parameters:
        X (array-like): Feature vectors (n_samples, n_features).

        Returns:
        array-like: Predicted values.
        """
        return self.model.predict(X)
    
    def cross_validate(self, X, y, cv=5, scoring='r2'):
        """
        Performs cross-validation on the model.

        Parameters:
        X (array-like): Feature vectors.
        y (array-like): Target values.
        cv (int): Number of cross-validation folds.
        scoring (str): Scoring metric (default: 'r2').

        Returns:
        dict: Cross-validation scores.
        """
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        self.cv_results_ = {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores)
        }
        return self.cv_results_
    
    def select_hyperparams(self, X, y, alphas=None, cv=5, scoring='r2'):
        """
        Performs hyperparameter selection using cross-validation (Grid Search).

        Parameters:
        X (array-like): Feature vectors.
        y (array-like): Target values.
        alphas (array-like): Alpha values to search. Default: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1].
        cv (int): Number of cross-validation folds.
        scoring (str): Scoring metric (default: 'r2').

        Returns:
        dict: Best parameters and corresponding score.
        """
        if alphas is None:
            alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        
        param_grid = {'alpha': alphas}
        grid_search = GridSearchCV(
            SGDRegressor(max_iter=self.max_iter, tol=self.tol, random_state=self.random_state),
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        grid_search.fit(X, y)
        
        self.best_alpha_ = grid_search.best_params_['alpha']
        self.model = SGDRegressor(
            alpha=self.best_alpha_,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X, y)
        
        return {
            'best_alpha': self.best_alpha_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def fit_and_evaluate(self, dataset, alphas=None, cv=5, val_size=0.2, scoring='r2'):
        """
        Fit and evaluate model on a ModelBrainDataset.
        
        Workflow:
        1. Split training data into train+val
        2. Select hyperparameters using cross-validation (no test data)
        3. Refit on full train+val
        4. Evaluate on held-out test set

        Parameters:
        dataset (ModelBrainDataset): Dataset with pre-separated train and test splits.
        alphas (array-like): Alpha values to search. If None, uses defaults.
        cv (int): Number of cross-validation folds for hyperparameter selection.
        val_size (float): Proportion of training data for validation.
        scoring (str): Scoring metric (default: 'r2').

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
        hp_results = self.select_hyperparams(
            X_train_val, y_train_val,
            alphas=alphas, cv=cv, scoring=scoring
        )
        
        # Evaluate on test set
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

