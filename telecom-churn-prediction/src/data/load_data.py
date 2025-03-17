import pandas as pd

def load_and_examine_data(file_path):
    """
    Load the data and perform initial examination
    """
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nData sample:")
    print(df.head())
    
    return df
