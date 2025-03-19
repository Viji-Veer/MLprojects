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

def prepare_modeling_data(df):
    """
    Prepare the data for modeling by encoding categorical variables
    and scaling numerical variables
    """
    print("\nPreparing data for modeling...")
    
    df_prep = df.copy()
    
    # Define the target variable (what we're trying to predict)
    y = df_prep['Churned_Binary']
    
    # Remove columns not needed for modeling
    columns_to_drop = ['Customer ID', 'Churned_Binary', 'Customer Status', 'Churn Category', 'Churn Reason',
                      'Zip Code', 'Latitude', 'Longitude', 'City']
    
    X = df_prep.drop(columns=columns_to_drop)
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\nCategorical columns: {len(categorical_cols)}")
    print(f"Numerical columns: {len(numerical_cols)}")
    
    # Create preprocessing pipelines for both categorical and numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine the preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocess the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get the feature names after one-hot encoding
    cat_feature_names = []
    
    if categorical_cols:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    
    all_feature_names = numerical_cols + cat_feature_names
    
    print(f"\nFinal processed dataset shape: {X_train_processed.shape}")
    
    return (X_train_processed, X_test_processed, y_train, y_test, 
            preprocessor, all_feature_names)
