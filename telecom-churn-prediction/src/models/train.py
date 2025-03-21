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
