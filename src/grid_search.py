import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier
import joblib  # for saving models

def train_and_evaluate_model(data, dataset_name, model_path, results_path):
    # Extract features and labels
    features = data.columns.difference(['PCT', 'Label'])
    X = data[features]
    y = data['Label']

    # Setup cross-validation
    cv = StratifiedKFold(n_splits=5)

    # Define the parameter grid
    param_grid = {
        'estimator__max_depth': [5, 10, 15],
        'n_estimators': [100, 300, 500, 700, 900],
        'learning_rate': [0.01, 0.1, 0.5, 1]
    }

    # Instantiate a RUSBoostClassifier with a DecisionTreeClassifier as the base estimator
    estimator = DecisionTreeClassifier(random_state=42)
    model = RUSBoostClassifier(estimator=estimator, random_state=42)

    # Setup the GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)

    # Perform the grid search
    grid_search.fit(X, y)

    # Best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Dataset: {dataset_name}")
    print("Best parameters:", best_params)
    print("Best score (ROC AUC):", best_score)

    # Save the model and best parameters
    joblib.dump(best_model, f'{model_path}/best_model_{dataset_name}.pkl')

    # Optionally, save best parameters in a text file or as a pickle
    with open(f'{results_path}/best_params_{dataset_name}.txt', 'w') as f:
        f.write(str(best_params))

    return best_model

def main():
        # Set paths
    model_path = 'C:\\Users\\menia\\Desktop\\PortefolioML\\Sepsis\\models'
    results_path = 'C:\\Users\\menia\\Desktop\\PortefolioML\\Sepsis\\results'
    # Load your datasets here
    data_P = pd.read_csv('../data/pct_data_sepsis_1_imputed.csv')
    data_N = pd.read_csv('../data/pct_data_sepsis_2_imputed.csv')

    # Train and evaluate on both datasets
    for data, name in zip([data_P, data_N], ['1', '2']):
        best_model = train_and_evaluate_model(data, name, model_path, results_path)

if __name__ == '__main__':
    main()
