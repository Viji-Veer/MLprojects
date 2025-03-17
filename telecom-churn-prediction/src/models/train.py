from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
import joblib

def handle_class_imbalance(X_train, y_train):
    """
    Handle class imbalance using SMOTE
    """
    print("\nHandling class imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Print class distribution before and after resampling
    print("Class distribution before resampling:")
    print(pd.Series(y_train).value_counts(normalize=True))
    
    print("\nClass distribution after resampling:")
    print(pd.Series(y_train_resampled).value_counts(normalize=True))
    
    return X_train_resampled, y_train_resampled

def hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning for Gradient Boosting
    """
    print("\nPerforming hyperparameter tuning for Gradient Boosting...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Use RandomizedSearchCV for more efficient tuning
    gb_clf = GradientBoostingClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        gb_clf, param_distributions=param_grid, 
        n_iter=20, cv=5, scoring='roc_auc',
        random_state=42, n_jobs=-1
    )
    
    # Fit the model
    random_search.fit(X_train, y_train)
    
    # Print best parameters and score
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best ROC AUC score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def train_final_model(X_train, y_train, model_save_path='models/best_churn_model.pkl'):
    """
    Train the final model and save to disk
    """
    # Handle class imbalance
    X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)
    
    # Hyperparameter tuning
    best_model = hyperparameter_tuning(X_train_resampled, y_train_resampled)
    
    # Train final model
    print("\nTraining final model with best parameters...")
    best_model.fit(X_train_resampled, y_train_resampled)
    
    # Save model
    joblib.dump(best_model, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return best_model
