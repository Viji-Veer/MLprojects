def fit_model_example(X_train, y_train, X_test, y_test):
    """
    Example of how to fit a model using the processed data
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
    
    print("\nFitting example models...")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred) 

        # For ROC AUC we need probability estimates
        try:
            y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class
            roc_auc = roc_auc_score(y_test, y_prob)
            roc_auc_display = f"{roc_auc:.4f}"
        except:
            roc_auc = None
            roc_auc_display = 'Not available'

        
        
        # Print metrics - fixed the formatting error
        print(f"{name} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc_display}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
       
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc
        }
    
    return results
