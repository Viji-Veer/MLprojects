def plot_feature_importance(model, feature_names, top_n=15):
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
    plt.savefig('feature_importance.png')
    plt.close()
    
    return feature_imp
