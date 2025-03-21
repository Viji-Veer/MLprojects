def optimize_threshold(model, X_test, y_test):
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
    plt.savefig('roc_curve.png')
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
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return optimal_threshold, y_pred_optimal


def evaluate_model_for_business(model, X_test, y_test, optimal_threshold=None):
    """
    Evaluate model with business-specific metrics
    """
    print("\nEvaluating model for business impact...")
    
    # Get probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Use optimal threshold if provided, otherwise use default 0.5
    threshold = optimal_threshold if optimal_threshold is not None else 0.5
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Calculate business metrics
    # Assuming:
    # - Cost of customer acquisition: $200
    # - Average monthly revenue per customer: $70
    # - Average customer lifetime (months): 24
    # - Retention offer cost: $50
    
    acquisition_cost = 200
    monthly_revenue = 70
    avg_lifetime = 24
    retention_offer_cost = 50
    
    # Calculate customer lifetime value (CLV)
    clv = monthly_revenue * avg_lifetime
    
    # Calculate costs/benefits
    true_positive_value = tp * (clv - retention_offer_cost)  # Correctly identified churners who we retain
    false_positive_cost = fp * retention_offer_cost          # Customers who wouldn't churn but got offer
    false_negative_cost = fn * clv                           # Missed churners - lost full CLV
    
    # Net benefit
    net_benefit = true_positive_value - false_positive_cost - false_negative_cost
    
    # Return on investment (ROI)
    total_cost = (tp + fp) * retention_offer_cost
    roi = (true_positive_value - total_cost) / total_cost if total_cost > 0 else 0
    
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"True Positives (correctly identified churners): {tp}")
    print(f"False Positives (unnecessarily offered retention): {fp}")
    print(f"False Negatives (missed churners): {fn}")
    print(f"True Negatives (correctly identified non-churners): {tn}")
    print(f"\nBusiness Impact:")
    print(f"Value from retained customers: ${true_positive_value:,.2f}")
    print(f"Cost of unnecessary retention offers: ${false_positive_cost:,.2f}")
    print(f"Cost of missed churners: ${false_negative_cost:,.2f}")
    print(f"Net benefit: ${net_benefit:,.2f}")
    print(f"ROI: {roi:.2%}")
    
    # Plot ROC curve with business-optimal threshold
    plt.figure(figsize=(10, 8))
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_prob):.4f})')
    plt.show()
    # Find the threshold that maximizes net benefit
    # In a real scenario, you'd calculate the net benefit for each threshold
    # Here we use the provided optimal threshold
    plt.scatter(fp/(fp+tn), tp/(tp+fn), marker='o', color='red', 
               label=f'Business-optimal threshold: {threshold:.4f}')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Business-Optimal Threshold')
    plt.legend()
    plt.savefig('business_roc_curve.png')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'net_benefit': net_benefit,
        'roi': roi
    }

