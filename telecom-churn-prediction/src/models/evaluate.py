import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.inspection import permutation_importance
import joblib

def plot_feature_importance(model, X_test, y_test, feature_names, top_n=15, save_path='reports/figures/feature_importance.png'):
    """
    Plot feature importance for a trained model
    """
    if hasattr(model, 'feature_importances_'):
        # For models with feature_importances_ attribute
        importances = model.feature_importances_
    else:
        # For models without feature_importances_ attribute, use permutation importance
        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        importances = perm_importance.importances_mean
    
    # Create a dataframe for visualization
    feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_imp = feature_imp.sort_values('Importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_imp)
    plt.title('Top Feature Importances')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return feature_imp

def optimize_threshold(model, X_test, y_test, save_path='reports/figures/roc_curve.png', 
                       conf_matrix_path='reports/figures/confusion_matrix.png', threshold_save_path='models/optimal_threshold.pkl'):
    """
    Find optimal threshold for classification based on ROC curve
    """
    print("\nOptimizing classification threshold...")
    # Get probabilities for the positive class
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    # Find the threshold that maximizes TPR - FPR (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_prob):.4f})')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', 
               label=f'Optimal threshold: {optimal_threshold:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Optimal Threshold')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    
    # Make predictions with the optimal threshold
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    
    # Print classification report with optimal threshold
    print("\nClassification report with optimal threshold:")
    print(classification_report(y_test, y_pred_optimal))
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_optimal)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Not Churned', 'Churned'],
               yticklabels=['Not Churned', 'Churned'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix with Optimal Threshold')
    plt.savefig(conf_matrix_path)
    plt.close()
    
    # Save the optimal threshold
    joblib.dump(optimal_threshold, threshold_save_path)
    print(f"Optimal threshold saved to {threshold_save_path}")
    
    return optimal_threshold, y_pred_optimal
