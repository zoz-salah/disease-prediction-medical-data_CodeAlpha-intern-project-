"""
Evaluation and visualization for classification models.
Includes confusion matrix, ROC curves, and feature importance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import joblib

def plot_confusion_matrix(y_true, y_pred, classes=None, normalize=False, save_path=None):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=classes if classes else ['Negative', 'Positive'],
                yticklabels=classes if classes else ['Negative', 'Positive'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.show()

def plot_roc_curves(models_dict, X_test, y_test, save_path=None):
    """
    Plot ROC curves for multiple models on same figure.
    models_dict: {model_name: trained_model}
    """
    plt.figure(figsize=(8, 6))
    
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    plt.show()

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    Plot feature importance for tree-based models (Random Forest, XGBoost)
    or coefficients for linear models.
    """
    plt.figure(figsize=(10, 6))
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        title = "Feature Importances"
    elif hasattr(model, 'coef_'):
        # For linear models, use absolute coefficients
        importances = np.abs(model.coef_).flatten()
        title = "Feature Coefficients (absolute)"
    else:
        print("Model does not have feature importances or coefficients.")
        return
    
    # Sort and select top_n
    indices = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in indices]
    values = importances[indices]
    
    plt.barh(range(len(indices)), values[::-1], align='center')
    plt.yticks(range(len(indices)), [names[i] for i in indices[::-1]])
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    plt.show()

def generate_classification_report(y_true, y_pred, target_names=None):
    """
    Print sklearn classification report.
    """
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred, target_names=target_names))

def evaluate_saved_model(model_path, X_test, y_test, feature_names=None, dataset_name='dataset'):
    """
    Load a saved model and run full evaluation.
    """
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")
    
    from models import evaluate_model
    metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)
    print("\nPerformance on test set:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Confusion matrix
    classes = ['Negative', 'Positive']
    plot_confusion_matrix(y_test, y_pred, classes=classes, normalize=False,
                          save_path=f'plots/{dataset_name}_cm.png')
    
    # ROC curve (if probabilities available)
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve')
        plt.legend()
        plt.savefig(f'plots/{dataset_name}_roc.png')
        plt.show()
    
    # Feature importance
    if feature_names is not None:
        plot_feature_importance(model, feature_names, save_path=f'plots/{dataset_name}_importance.png')
    
    return metrics

# ----------------------------------------------------------------------
# Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import load_dataset
    X_train, X_test, y_train, y_test, feature_names = load_dataset('heart')
    from models import train_model
    model = train_model('random_forest', X_train, y_train)
    from models import evaluate_model
    metrics, y_pred, _ = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model, feature_names)
  
