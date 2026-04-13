import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import os

def save_model(model, filepath):
    """Save model using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load model using joblib."""
    return joblib.load(filepath)

def plot_class_distribution(y, title='Class Distribution'):
    """Bar plot of class counts."""
    plt.figure(figsize=(6,4))
    counts = pd.Series(y).value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.show()

def create_submission_file(ids, predictions, filename='submission.csv'):
    """Save predictions for competition-style output."""
    pd.DataFrame({'id': ids, 'prediction': predictions}).to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
