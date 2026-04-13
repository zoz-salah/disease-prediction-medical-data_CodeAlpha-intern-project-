import argparse
import os
import pandas as pd
from src.data_loader import load_dataset
from src.models import train_all_models, hyperparameter_tuning
from src.evaluate import evaluate_saved_model
import joblib

def main():
    parser = argparse.ArgumentParser(description='Disease Prediction from Medical Data')
    parser.add_argument('--mode', type=str, default='train_all',
                        choices=['train_all', 'tune', 'evaluate', 'predict'],
                        help='Operation mode')
    parser.add_argument('--dataset', type=str, default='heart',
                        choices=['heart', 'diabetes', 'breast_cancer'],
                        help='Medical dataset to use')
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['logistic_regression', 'svm', 'random_forest', 'xgboost'],
                        help='Model type (for tune/evaluate/predict)')
    parser.add_argument('--csv', type=str, help='CSV file for prediction')
    parser.add_argument('--output', type=str, help='Output file for predictions')
    parser.add_argument('--interactive', action='store_true', help='Interactive prediction')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    X_train, X_test, y_train, y_test, feature_names = load_dataset(args.dataset)
    
    if args.mode == 'train_all':
        # Train all four models and evaluate
        results_df = train_all_models(X_train, y_train, X_test, y_test, save_dir='models')
        # Save results to CSV
        results_df.to_csv(f'plots/{args.dataset}_model_comparison.csv')
        print(f"Results saved to plots/{args.dataset}_model_comparison.csv")
        
    elif args.mode == 'tune':
        # Hyperparameter tuning for a specific model
        best_model = hyperparameter_tuning(args.model, X_train, y_train)
        # Save tuned model
        tuned_path = f'models/{args.model}_tuned.pkl'
        joblib.dump(best_model, tuned_path)
        print(f"Tuned model saved to {tuned_path}")
        # Evaluate on test set
        from src.models import evaluate_model
        metrics, _, _ = evaluate_model(best_model, X_test, y_test)
        print("Test performance of tuned model:")
        for k,v in metrics.items():
            print(f"  {k}: {v:.4f}")
            
    elif args.mode == 'evaluate':
        # Evaluate a specific saved model
        model_path = f'models/{args.model}.pkl'
        if not os.path.exists(model_path):
            model_path = f'models/{args.model}_tuned.pkl'  # try tuned
        evaluate_saved_model(model_path, X_test, y_test, feature_names, args.dataset)
        
    elif args.mode == 'predict':
        # Prediction on new data
        from src.predict import load_model_and_scaler, predict_from_csv, interactive_prediction
        model, scaler = load_model_and_scaler(args.model, args.dataset)
        if args.csv:
            predict_from_csv(model, scaler, args.csv, list(feature_names), args.output)
        elif args.interactive:
            interactive_prediction(model, scaler, list(feature_names))
        else:
            print("Please provide --csv or --interactive for prediction.")

if __name__ == "__main__":
    main()
