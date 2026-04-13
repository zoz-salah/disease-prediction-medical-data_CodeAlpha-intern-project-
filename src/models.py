"""
Training multiple classification models on medical datasets.
Supports: Logistic Regression, SVM, Random Forest, XGBoost.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score

# Dictionary of model names and their constructors with default hyperparameters
MODELS = {
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
    'svm': SVC(probability=True, random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'xgboost': XGBClassifier(eval_metric='logloss', random_state=42)
}

def train_model(model_name, X_train, y_train, hyperparams=None):
    """
    Train a single model with given hyperparameters.
    
    Args:
        model_name: key from MODELS dict
        X_train: training features (numpy array)
        y_train: training labels
        hyperparams: dict of parameters to override defaults
    
    Returns:
        trained model
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}")
    
    model = MODELS[model_name].__class__()  # new instance
    # Set default params from MODELS
    model.set_params(**MODELS[model_name].get_params())
    if hyperparams:
        model.set_params(**hyperparams)
    
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set and return metrics dict.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    
    return metrics, y_pred, y_proba

def train_all_models(X_train, y_train, X_test, y_test, save_dir='models'):
    """
    Train all four models with default parameters and save them.
    Returns a DataFrame with test performance.
    """
    os.makedirs(save_dir, exist_ok=True)
    results = []
    
    for name in MODELS.keys():
        print(f"\n{'='*50}")
        print(f"Model: {name}")
        model = train_model(name, X_train, y_train)
        
        # Save model
        model_path = os.path.join(save_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        # Evaluate
        metrics, _, _ = evaluate_model(model, X_test, y_test)
        metrics['model'] = name
        results.append(metrics)
        
        # Print metrics
        for k, v in metrics.items():
            if k != 'model':
                print(f"  {k}: {v:.4f}")
    
    # Summary DataFrame
    results_df = pd.DataFrame(results)
    results_df.set_index('model', inplace=True)
    print("\n" + "="*50)
    print("Performance Summary:")
    print(results_df.round(4))
    
    return results_df

def hyperparameter_tuning(model_name, X_train, y_train, param_grid=None, cv=5):
    """
    Perform GridSearchCV for a specific model.
    Returns best model.
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'")
    
    # Default param grids for common models
    default_grids = {
        'logistic_regression': {'C': [0.1, 1, 10]},
        'svm': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
        'random_forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
        'xgboost': {'n_estimators': [50, 100], 'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1]}
    }
    
    grid = param_grid if param_grid else default_grids[model_name]
    base_model = MODELS[model_name].__class__()  # fresh instance
    
    print(f"Starting GridSearchCV for {model_name} with grid: {grid}")
    gs = GridSearchCV(base_model, grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    
    print(f"Best parameters: {gs.best_params_}")
    print(f"Best CV score (ROC AUC): {gs.best_score_:.4f}")
    
    return gs.best_estimator_

# ----------------------------------------------------------------------
# Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import load_dataset
    X_train, X_test, y_train, y_test, _ = load_dataset('heart')
    train_all_models(X_train, y_train, X_test, y_test)
