
import pandas as pd
import numpy as np

def clean_data(df):
    """
    Clean the data by handling missing values and fixing data types
    """
        
    # Create a copy to avoid modifying the original dataframe
    df_clean = df.copy()
    
    # Fix any negative values in monetary columns (observed -4 in Monthly Charge)
    monetary_columns = ['Monthly Charge', 'Total Charges', 'Total Refunds', 
                        'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue']
    
    for col in monetary_columns:
        if (df_clean[col] < 0).any():
            print(f"Fixing negative values in {col}")
            # Replace negative values with 0 or median based on business context
            df_clean[col] = np.where(df_clean[col] < 0, df_clean[col].median(), df_clean[col])
    
    # Convert Zip Code to string to preserve leading zeros
    df_clean['Zip Code'] = df_clean['Zip Code'].astype(str)
    
    # Handle missing values in Churn Category and Churn Reason for customers who stayed
    # We'll create a new category called "No Churn" for these customers
    df_clean['Churn Category'] = df_clean['Churn Category'].fillna('No Churn')
    df_clean['Churn Reason'] = df_clean['Churn Reason'].fillna('No Churn')
    
    return df_clean
