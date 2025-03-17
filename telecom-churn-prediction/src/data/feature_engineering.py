import pandas as pd
import numpy as np

def create_binary_target(df):
    """
    Create binary target variable from Customer Status
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    
    Returns:
    pandas.DataFrame: DataFrame with binary target
    """
    df_featured = df.copy()
    df_featured['Churned_Binary'] = np.where(df_featured['Customer Status'] == 'Churned', 1, 0)
    return df_featured

def create_tenure_features(df):
    """
    Create tenure-related features
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    
    Returns:
    pandas.DataFrame: DataFrame with additional tenure features
    """
    df_featured = df.copy()
    
    # Calculate tenure in years
    df_featured['Tenure_Years'] = df_featured['Tenure in Months'] / 12
    
    # Create tenure bins
    df_featured['Tenure_Group'] = pd.cut(
        df_featured['Tenure in Months'],
        bins=[0, 12, 24, 36, 60, 100],
        labels=['0-1 year', '1-2 years', '2-3 years', '3-5 years', '5+ years']
    )
    
    return df_featured

def create_demographic_features(df):
    """
    Create demographic-related features
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    
    Returns:
    pandas.DataFrame: DataFrame with additional demographic features
    """
    df_featured = df.copy()
    
    # Create binned age groups
    df_featured['Age_Group'] = pd.cut(
        df_featured['Age'], 
        bins=[0, 30, 45, 60, 100], 
        labels=['Under 30', '30-45', '46-60', 'Over 60']
    )
    
    # Family status
    df_featured['Has_Family'] = np.where(
        (df_featured['Married'] == 'Yes') | (df_featured['Number of Dependents'] > 0),
        1, 0
    )
    
    return df_featured

def create_service_features(df):
    """
    Create service-related features
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    
    Returns:
    pandas.DataFrame: DataFrame with additional service features
    """
    df_featured = df.copy()
    
    # Create a flag for premium services
    premium_services = ['Online Security', 'Online Backup', 'Device Protection Plan', 
                      'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music']
    
    # Count how many premium services each customer has
    df_featured['Premium_Services_Count'] = 0
    for service in premium_services:
        df_featured['Premium_Services_Count'] += np.where(df_featured[service] == 'Yes', 1, 0)
    
    # Flag for high data users
    df_featured['High_Data_User'] = np.where(df_featured['Avg Monthly GB Download'] > 50, 1, 0)
    
    return df_featured

def create_financial_features(df):
    """
    Create financial-related features
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    
    Returns:
    pandas.DataFrame: DataFrame with additional financial features
    """
    df_featured = df.copy()
    
    # Monthly spend ratio (Monthly Charge / Tenure)
    # Avoid division by zero
    df_featured['Monthly_Spend_Ratio'] = df_featured['Monthly Charge'] / df_featured['Tenure in Months'].replace(0, 1)
    
    # Total charges per month
    df_featured['Average_Monthly_Charges'] = df_featured['Total Charges'] / df_featured['Tenure in Months'].replace(0, 1)
    
    return df_featured

def engineer_features(df):
    """
    Apply all feature engineering steps
    
    Parameters:
    df (pandas.DataFrame): Cleaned DataFrame
    
    Returns:
    pandas.DataFrame: DataFrame with all engineered features
    """
    df = create_binary_target(df)
    df = create_tenure_features(df)
    df = create_demographic_features(df)
    df = create_service_features(df)
    df = create_financial_features(df)
    return df
