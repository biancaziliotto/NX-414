import numpy as np
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import h5py

class ModelBrainDataset():
    """
    Dataset class for handling model-based encoding tasks.
    Loads model activations for given stimuli and pairs them with neural responses.
    Assumes train and test sets are already separated.
    """
    def __init__(self, y_train, y_test, stimuli_train, stimuli_test, model_name, activations_path):
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
        self.activations_path = activations_path
        
        # Load activations
        self.X_train = self._load_activations(stimuli_train, model_name, activations_path)
        self.X_test = self._load_activations(stimuli_test, model_name, activations_path)
        
        self.X_val = None
        self.y_val = None
        self.X_train_split = None
        self.y_train_split = None

    def _load_activations(self, stimuli_ids, model_name, path):
        """
        Loads model activations for the given stimuli.

        Parameters:
        stimuli_ids (array-like): Stimuli identifiers/indices.
        model_name (str): Name of the model.
        path_template (str): Template path for loading activations.

        Returns:
        array-like: Stacked activations (n_samples, n_features).
        """
        activations_list = []
        for stim_id in stimuli_ids:

            # Load activation (placeholder - adjust based on your file format)
            with h5py.File(path, "r") as f:
                model_layers = list(f['features'].keys())
                features = f['features'][model_layers[0]][:]
                ids = list(f['ids'][:])

            for stim_id in stimuli_ids:
                activation = features[ids.index(stim_id),:]
            activations_list.append(activation)
        
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

