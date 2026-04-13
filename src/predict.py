import numpy as np
import pandas as pd
import joblib
import os
import argparse

def load_model_and_scaler(model_name, dataset_name):
    """
    Load trained model and associated scaler.
    """
    model_path = f'models/{model_name}.pkl'
    scaler_path = f'models/{dataset_name}_scaler.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    return model, scaler

def predict_single(model, scaler, features_dict, feature_order):
    """
    Make prediction for a single patient.
    
    Args:
        model: trained sklearn model
        scaler: StandardScaler fitted on training data
        features_dict: dict with feature values (can be incomplete? better exact)
        feature_order: list of feature names in correct order
    
    Returns:
        prediction (0/1), probability of positive class
    """
    # Convert dict to ordered list
    try:
        feature_values = [features_dict[feat] for feat in feature_order]
    except KeyError as e:
        raise ValueError(f"Missing feature: {e}. Required features: {feature_order}")
    
    X = np.array(feature_values).reshape(1, -1)
    
    # Scale if scaler provided
    if scaler is not None:
        X = scaler.transform(X)
    
    pred_class = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0, 1] if hasattr(model, "predict_proba") else None
    
    return pred_class, pred_proba

def predict_from_csv(model, scaler, csv_path, feature_order, output_path=None):
    """
    Batch prediction from CSV file.
    """
    df = pd.read_csv(csv_path)
    # Ensure all required features exist
    missing = set(feature_order) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    
    X = df[feature_order].values
    if scaler:
        X = scaler.transform(X)
    
    df['Predicted_Class'] = model.predict(X)
    if hasattr(model, "predict_proba"):
        df['Probability'] = model.predict_proba(X)[:, 1]
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    else:
        print(df.head())
    
    return df

def interactive_prediction(model, scaler, feature_order):
    """
    Prompt user for feature values one by one.
    """
    print("\n" + "="*50)
    print("Interactive Disease Prediction")
    print("Enter patient data when prompted:")
    features_dict = {}
    for feat in feature_order:
        while True:
            try:
                val = float(input(f"  {feat}: "))
                features_dict[feat] = val
                break
            except ValueError:
                print("    Please enter a numeric value.")
    
    pred_class, proba = predict_single(model, scaler, features_dict, feature_order)
    
    print("\n" + "-"*30)
    if pred_class == 1:
        print(f"⚠️  High risk of disease detected.")
        if proba is not None:
            print(f"   Probability: {proba:.2%}")
    else:
        print(f"✅ Low risk of disease.")
        if proba is not None:
            print(f"   Probability of disease: {proba:.2%}")
    print("-"*30)
    
    return pred_class, proba

# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict disease from patient data.')
    parser.add_argument('--model', type=str, required=True,
                        choices=['logistic_regression', 'svm', 'random_forest', 'xgboost'],
                        help='Model to use')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['heart', 'diabetes', 'breast_cancer'],
                        help='Dataset the model was trained on')
    parser.add_argument('--csv', type=str, help='CSV file for batch prediction')
    parser.add_argument('--output', type=str, help='Output CSV file for predictions')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Load feature order from dataset
    from data_loader import load_dataset
    # We only need feature names, but load_dataset returns data. Let's get names directly.
    if args.dataset == 'heart':
        from data_loader import HEART_COLUMNS
        feature_names = HEART_COLUMNS[:-1]  # exclude target
    elif args.dataset == 'diabetes':
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    elif args.dataset == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer
        feature_names = list(load_breast_cancer().feature_names)
    
    # Load model and scaler
    model, scaler = load_model_and_scaler(args.model, args.dataset)
    print(f"Loaded model: {args.model}")
    
    if args.csv:
        predict_from_csv(model, scaler, args.csv, feature_names, args.output)
    elif args.interactive:
        interactive_prediction(model, scaler, feature_names)
    else:
        print("Please specify --csv for batch prediction or --interactive for manual input.")
