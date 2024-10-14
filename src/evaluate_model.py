import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # Eables the experimental features
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor
import joblib

def load_and_preprocess_data(file_path, columns):
    """Load data and filter specific columns."""
    data = pd.read_csv(file_path)
    return data[columns]

def knn_impute(data):
    """Impute missing values using KNN."""
    data = pd.get_dummies(data, columns=['sex'], drop_first=True)
    imputer = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=5), random_state=0, max_iter=10)
    imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return imputed_data

def predict_and_save_results(data, model, results_path):
    """Predict using the loaded model and save the results."""
    # Extract features for prediction (exclude 'Label' and 'PCT')
    features = data.columns.difference(['PCT', 'Label'])
    X = data[features]
    y = data['Label']
    
    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]  # probabilities for the positive class
    
    # Create a DataFrame to hold results
    results = pd.DataFrame({'Actual': y, 'Predicted': predictions, 'Probability': probabilities})
    
    # Save results
    results_file_path = f'{results_path}\\classification_results.csv'
    results.to_csv(results_file_path, index=False)
    
    print(f"Results saved to {results_file_path}")
    return results

def main():
    data_path = 'C:\\Users\\menia\\Desktop\\PortefolioML\\Sepsis\\data\\pct_data_sepsis_intermediate.csv'
    model_path = 'C:\\Users\\menia\\Desktop\\PortefolioML\\Sepsis\\models\\trained_model_2.pkl'
    results_path = 'C:\\Users\\menia\\Desktop\\PortefolioML\\Sepsis\\results'
    
    # Columns to be used from the dataset
    columns_subdataset2 = ['age', 'sex', 'HB', 'MCV', 'THROMB', 'ERY', 'LEUKO', 'LYMABS', 'LYMPHO', 'EOSABS', 'BASOAB', 'HK', 'RDW-SD', 'PCT', 'Label']
    
    # Load and preprocess the data
    data = load_and_preprocess_data(data_path, columns_subdataset2)
    
    # Impute missing values using KNN
    imputed_data = knn_impute(data)
    
    # Load the pre-trained model
    model = joblib.load(model_path)
    
    # Predict and save results
    results = predict_and_save_results(imputed_data, model, results_path)
    
    # Optionally print the results to the console or perform additional analysis
    print(results.head())  # Display the first few rows of the results

if __name__ == '__main__':
    main()
