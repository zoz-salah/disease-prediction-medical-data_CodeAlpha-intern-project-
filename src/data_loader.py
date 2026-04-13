"""
Dataset loading and preprocessing for UCI medical datasets.
Supports:
- Heart Disease (Cleveland)
- Diabetes (Pima Indians)
- Breast Cancer Wisconsin

All datasets are fetched from online sources and cached locally.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_breast_cancer, load_diabetes
import urllib.request

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# URLs for datasets not available in sklearn
HEART_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
HEART_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def download_file(url, local_path):
    """Download a file if not already present."""
    if not os.path.exists(local_path):
        print(f"Downloading {url} to {local_path}...")
        urllib.request.urlretrieve(url, local_path)
    return local_path

# ----------------------------------------------------------------------
# Dataset loaders
# ----------------------------------------------------------------------
def load_heart_disease():
    """
    Load Cleveland Heart Disease dataset from UCI.
    Target: 0 = no disease, 1-4 = disease (we binarize to 0/1).
    """
    local_file = os.path.join(DATA_DIR, 'heart.csv')
    download_file(HEART_URL, local_file)
    
    df = pd.read_csv(local_file, header=None, names=HEART_COLUMNS, na_values='?')
    
    # Drop rows with missing values (few)
    df = df.dropna()
    
    # Binarize target: 0 -> 0 (no disease), 1,2,3,4 -> 1 (disease)
    df['target'] = (df['target'] > 0).astype(int)
    
    # Features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Encode categorical features (none actually, all numeric)
    # But some like cp, restecg, slope, ca, thal are categorical integers
    
    print(f"Heart Disease dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Positive class ratio: {y.mean():.2f}")
    
    return X, y, df.columns[:-1]

def load_diabetes_pima():
    """
    Load Pima Indians Diabetes dataset from UCI (via local file if not already).
    We'll download from a reliable source or use sklearn's load_diabetes (regression) - no.
    Actually sklearn has no classification diabetes; we fetch from GitHub mirror.
    """
    local_file = os.path.join(DATA_DIR, 'pima-indians-diabetes.csv')
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    
    if not os.path.exists(local_file):
        print(f"Downloading Pima Indians Diabetes dataset...")
        urllib.request.urlretrieve(url, local_file)
    
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    df = pd.read_csv(local_file, names=columns)
    
    # Some zeros in features like Glucose, BloodPressure are biologically impossible -> replace with NaN and impute median
    zero_not_possible = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_not_possible:
        df[col] = df[col].replace(0, np.nan)
    
    # Impute with median
    for col in zero_not_possible:
        df[col].fillna(df[col].median(), inplace=True)
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    print(f"Diabetes dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Positive class ratio: {y.mean():.2f}")
    
    return X, y, X.columns

def load_breast_cancer_wisconsin():
    """
    Load Breast Cancer Wisconsin dataset from sklearn.
    """
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target  # 0=malignant, 1=benign (we'll keep as is)
    feature_names = data.feature_names
    
    print(f"Breast Cancer dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Positive class (benign) ratio: {y.mean():.2f}")
    
    return X, y, feature_names

# ----------------------------------------------------------------------
# Unified loader
# ----------------------------------------------------------------------
def load_dataset(name='heart', test_size=0.2, random_state=42, scale=True):
    """
    Load and preprocess a dataset.
    
    Args:
        name: 'heart', 'diabetes', or 'breast_cancer'
        test_size: proportion for test split
        random_state: seed
        scale: whether to standardize features
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    loaders = {
        'heart': load_heart_disease,
        'diabetes': load_diabetes_pima,
        'breast_cancer': load_breast_cancer_wisconsin
    }
    
    if name not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(loaders.keys())}")
    
    X, y, feature_names = loaders[name]()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Feature scaling
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Store scaler for later use in prediction
        import joblib
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, f'models/{name}_scaler.pkl')
        print(f"Scaler saved to models/{name}_scaler.pkl")
        
    print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, feature_names

# ----------------------------------------------------------------------
# Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    for ds in ['heart', 'diabetes', 'breast_cancer']:
        print(f"\n--- Loading {ds} ---")
        X_tr, X_te, y_tr, y_te, names = load_dataset(ds, test_size=0.2)
        print(f"Feature names: {list(names)[:5]}...")
