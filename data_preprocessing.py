import pandas as pd

def load_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)
    return df

def handle_missing_values(df):
    # Replace missing values with column mean
    df = df.fillna(df.mean())
    return df

def encode_categorical(df):
    # Encoding 'sex' column (example)
    df['sex'] = df['sex'].map({0: 'Female', 1: 'Male'})
    df['sex'] = df['sex'].map({'Female': 0, 'Male': 1})
    return df
