import pandas as pd
import numpy as np

def load_and_examine_data(file_path):
    """
    Load the data and perform initial examination
    
    Parameters:
    file_path (str): Path to the raw data file
    
    Returns:
    pandas.DataFrame: Raw data loaded into DataFrame
    """
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    return df

def clean_data(df):
    """
    Clean the data by handling missing values and fixing data types
    
    Parameters:
    df (pandas.DataFrame): Raw data DataFrame
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    print("\nCleaning data...")
    
    # Create a copy to avoid modifying the original dataframe
    df_clean = df.copy()
    
    # Fix any negative values in monetary columns (observed -4 in Monthly Charge)
    monetary_columns = ['Monthly Charge', 'Total Charges', 'Total Refunds', 
                      'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue']
    
    for col in monetary_columns:
        if (df_clean[col] < 0).any():
            print(f"Fixing negative values in {col}")
            # Replace negative values with median
            df_clean[col] = np.where(df_clean[col] < 0, df_clean[col].median(), df_clean[col])
    
    # Convert Zip Code to string to preserve leading zeros
    df_clean['Zip Code'] = df_clean['Zip Code'].astype(str)
    
    # Handle missing values in Churn Category and Churn Reason for customers who stayed
    df_clean['Churn Category'] = df_clean['Churn Category'].fillna('No Churn')
    df_clean['Churn Reason'] = df_clean['Churn Reason'].fillna('No Churn')
    
    return df_clean

def load_and_clean_data(file_path):
    """
    Load and clean the data in one function
    
    Parameters:
    file_path (str): Path to the raw data file
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    df = load_and_examine_data(file_path)
    df_clean = clean_data(df)
    return df_clean
