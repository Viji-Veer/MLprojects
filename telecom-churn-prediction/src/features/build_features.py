import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def feature_engineering(df):
    """
    Create new features that might be useful for modeling
    """
        
    df_featured = df.copy()
    
    # Create a binary churn indicator
    df_featured['Churned_Binary'] = np.where(df_featured['Customer Status'] == 'Churned', 1, 0)
    
    # Calculate tenure in years
    df_featured['Tenure_Years'] = df_featured['Tenure in Months'] / 12
    
    # Create binned age groups
    df_featured['Age_Group'] = pd.cut(
        df_featured['Age'], 
        bins=[0, 30, 45, 60, 100], 
        labels=['Under 30', '30-45', '46-60', 'Over 60']
    )
    
    # Create a flag for premium services
    premium_services = ['Online Security', 'Online Backup', 'Device Protection Plan', 
                        'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music']
    
    # Count how many premium services each customer has
    df_featured['Premium_Services_Count'] = 0
    for service in premium_services:
        df_featured['Premium_Services_Count'] += np.where(df_featured[service] == 'Yes', 1, 0)
    
    # Flag for high data users
    df_featured['High_Data_User'] = np.where(df_featured['Avg Monthly GB Download'] > 50, 1, 0)
    
    # Monthly spend ratio (Monthly Charge / Tenure)
    # Avoid division by zero
    df_featured['Monthly_Spend_Ratio'] = df_featured['Monthly Charge'] / df_featured['Tenure in Months'].replace(0, 1)
    
    return df_featured

