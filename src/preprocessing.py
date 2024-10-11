import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def load_and_concatenate_datasets(file_paths):
    """Load multiple datasets and concatenate them into a single DataFrame."""
    dataframes = [pd.read_csv(f, sep=';') for f in file_paths]
    return pd.concat(dataframes, ignore_index=True)

def define_age(df):
    current_year = 2024
    df['age'] = current_year - df['year_of_birth']
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset by forward filling."""
    df.ffill(inplace=True)
    return df

def replace_invalid_values(df):
    """Replace invalid values with NaN."""
    df.replace(-1, np.nan, inplace=True)
    return df

def add_label_column(df):
    """Add a 'Label' column based on 'PCT' values."""
    df['Label'] = np.where(df['PCT'] > 2, 1, 0)
    return df

def preprocess_datasets(df):
    """Apply preprocessing and categorization."""
    # Define a function that categorizes sepsis based on PCT values
    def categorize_sepsis(pct_value):
        if pct_value > 2:
            return 'P'  # Positive, sepsis likely
        elif pct_value >= 0.1:
            return 'I'  # Intermediate
        else:
            return 'N'  # Negative
    
    # Apply categorization
    df['sepsis_cat'] = df['PCT'].apply(categorize_sepsis)
    
    # Filter data based on 'sex' and transform it
    df = df[df['sex'].isin(['m', 'f'])]
    df['sex'] = df['sex'].replace({'f': 'W', 'm': 'M'})

    # Drop unwanted columns
    columns_to_drop = ['%h-ERY', 'NOR-Ge', 'XNRBC', 'Unnamed: 0', 'year_of_birth']
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    
    # Split data into subsets based on sepsis category
    data_P = df[df['sepsis_cat'] == 'P']
    data_N = df[df['sepsis_cat'] == 'N']
    data_I = df[df['sepsis_cat'] == 'I']
    
    return data_P, data_N, data_I

def save_preprocessed_data(df, file_path):
    """Save the preprocessed data to a CSV file."""
    df.to_csv(file_path, index=False)

def main():
    file_paths = [
        "../data/pct_data_2018.csv",
        "../data/pct_data_2019.csv",
        "../data/pct_data_2020.csv",
        "../data/pct_data_2021.csv",
        "../data/pct_data_2022.csv",
        "../data/pct_data_2023.csv",
        "../data/pct_data_2024.csv"
    ]
    # Load and concatenate data
    df = load_and_concatenate_datasets(file_paths)
    
    # Preprocess data
    df = define_age(df)
    df = handle_missing_values(df)
    df = replace_invalid_values(df)
    df = add_label_column(df)
    
    
    # Further preprocess and split data
    data_P, data_N, data_I = preprocess_datasets(df)
    
    # Save subsets
    save_preprocessed_data(data_P, '../data/pct_data_sepsis_positive.csv')
    save_preprocessed_data(data_N, '../data/pct_data_sepsis_negative.csv')
    save_preprocessed_data(data_I, '../data/pct_data_sepsis_intermediate.csv')

if __name__ == '__main__':
    main()
