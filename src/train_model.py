import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier
import joblib  # for saving models
import ast  # for converting string representation of dictionary into dictionary

def read_best_params(file_path):
    with open(file_path, 'r') as file:
        params = file.read()
        best_params = ast.literal_eval(params)  # safely evaluate string as a Python expression
    return best_params

def train_with_best_params(data, dataset_name, model_path, best_params):
    # Extract features and labels
    features = data.columns.difference(['PCT', 'Label'])
    X = data[features]
    y = data['Label']

    # Setup cross-validation
    cv = StratifiedKFold(n_splits=5)

    # Instantiate the model with best parameters
    estimator = DecisionTreeClassifier(max_depth=best_params['estimator__max_depth'], random_state=42)
    model = RUSBoostClassifier(
        estimator=estimator,
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        random_state=42
    )

    # Fit the model
    model.fit(X, y)

    # Save the trained model
    joblib.dump(model, f'{model_path}/trained_model_{dataset_name}.pkl')  # Save model

    print(f"Model trained and saved for dataset: {dataset_name}")

    return model

def main():
    # Set paths
    model_path = 'C:\\Users\\menia\\Desktop\\PortefolioML\\Sepsis\\models'
    results_path = 'C:\\Users\\menia\\Desktop\\PortefolioML\\Sepsis\\results'
    # Load your datasets here
    data_P = pd.read_csv('../data/pct_data_sepsis_1_imputed.csv')
    data_N = pd.read_csv('../data/pct_data_sepsis_2_imputed.csv')

    # Extract best parameters from text files
    best_params_P = read_best_params(f'{results_path}\\best_params_1.txt')
    best_params_N = read_best_params(f'{results_path}\\best_params_2.txt')

    # Train with best parameters on both datasets
    train_with_best_params(data_P, '1', model_path, best_params_P)
    train_with_best_params(data_N, '2', model_path, best_params_N)

if __name__ == '__main__':
    main()
