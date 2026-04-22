import numpy as np
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

class ModelBrainDataset():
    """
    Dataset class for handling model features and neural responses.
    Supports train/test/validation splits and cross-validation.
    Allows for different models and different neural datasets to be handled.
    """
    def __init__(self, model_features, neural_responses):
        """
        Initializes the dataset with model features and neural responses.

        Parameters:
        model_features (array-like): Model feature vectors (n_samples, n_features).
        neural_responses (array-like): Neural response values (n_samples, n_uni).
        """
        self.model_features = model_features
        self.neural_responses = neural_responses

        self.X_train = None
        self.X_test = None
        self.X_val = None

        self.y_train = None
        self.y_test = None
        self.y_val = None

    def get_data(self):
        """
        Returns the model features and neural responses.

        Returns:
        tuple: (model_features, neural_responses)
        """
        return self.model_features, self.neural_responses
    
    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Splits data into train, validation, and test sets.

        Parameters:
        test_size (float): Proportion of data for test set.
        val_size (float): Proportion of data for validation set.
        random_state (int): Random seed for reproducibility.
        """
        # Split into train+val and test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.model_features, self.neural_responses,
            test_size=test_size, random_state=random_state
        )
        
        # Split train+val into train and val
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted, random_state=random_state
        )
    
    def get_splits(self):
        """
        Returns the train/val/test splits.

        Returns:
        dict: Dictionary with 'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'.
        """
        return {
            'X_train': self.X_train,
            'X_val': self.X_val,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_val': self.y_val,
            'y_test': self.y_test
        }
    
    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
        int: Number of samples.
        """
        return len(self.model_features)

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

