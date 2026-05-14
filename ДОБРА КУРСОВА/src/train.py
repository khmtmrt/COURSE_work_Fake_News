"""
Model training with GridSearch (2d)
"""

import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from typing import Tuple, Dict, Any
from src.config import RANDOM_STATE, TEST_SIZE, MODELS_PATH


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.
    
    Args:
        X: Features
        y: Labels
        test_size: Proportion of test set
        random_state: Random state
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def get_model_params(model_name: str) -> Tuple[Any, Dict]:
    """
    Get model and parameter grid for GridSearch.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model instance and parameter grid
    """
    if model_name == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        }
    elif model_name == 'random_forest':
        model = RandomForestClassifier(random_state=RANDOM_STATE)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
    elif model_name == 'svm':
        model = SVC(random_state=RANDOM_STATE)
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model, param_grid


def train_model_with_gridsearch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = 'logistic_regression',
    cv: int = 5,
    n_jobs: int = -1
) -> GridSearchCV:
    """
    Train model with GridSearch.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_name: Name of the model
        cv: Number of cross-validation folds
        n_jobs: Number of parallel jobs
        
    Returns:
        Fitted GridSearchCV object
    """
    model, param_grid = get_model_params(model_name)
    
    print(f"Training {model_name} with GridSearch...")
    print(f"Parameter grid: {param_grid}")
    
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search


def save_model(model: Any, model_name: str) -> str:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        model_name: Name for the saved model
        
    Returns:
        Path to saved model
    """
    filepath = MODELS_PATH / f"{model_name}.joblib"
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
    return str(filepath)


def load_model(filepath: str) -> Any:
    """
    Load model from disk.
    
    Args:
        filepath: Path to model file
        
    Returns:
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
