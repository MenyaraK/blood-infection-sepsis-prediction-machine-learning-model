import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor

def load_datasets():
    data_path = '../data'
    data_positive = pd.read_csv(f'{data_path}/pct_data_sepsis_positive.csv')
    data_negative = pd.read_csv(f'{data_path}/pct_data_sepsis_negative.csv')
    return pd.concat([data_positive, data_negative])

def select_and_split_data(data):
    columns_subdataset1 = ['age', 'sex', 'HB', 'MCV', 'THROMB', 'ERY', 'LEUKO', 'PCT', 'Label']
    columns_subdataset2 = ['age', 'sex', 'HB', 'MCV', 'THROMB', 'ERY', 'LEUKO', 'LYMABS', 'LYMPHO', 'EOSABS', 'BASOAB', 'HK', 'RDW-SD', 'PCT', 'Label']
    
    subdataset1 = data[columns_subdataset1]
    subdataset2 = data[columns_subdataset2]
    
    return subdataset1, subdataset2

def knn_impute(data):
    """Impute missing values using KNN."""
    data = pd.get_dummies(data, columns=['sex'], drop_first=True)
    imputer = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=5), random_state=0, max_iter=10)
    imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return imputed_data

def save_preprocessed_data(data, filename):
    data.to_csv(filename, index=False)

def main():
    data = load_datasets()
    subdataset1, subdataset2= select_and_split_data(data)
    
    data_1_imputed = knn_impute(subdataset1)
    data_2_imputed = knn_impute(subdataset2)
    
    save_preprocessed_data(data_1_imputed, '../data/pct_data_sepsis_1_imputed.csv')
    save_preprocessed_data(data_2_imputed, '../data/pct_data_sepsis_2_imputed.csv')

if __name__ == '__main__':
    main()
