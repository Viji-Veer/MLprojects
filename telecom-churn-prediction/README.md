# Telecom Customer Churn Prediction

## Overview
This project uses machine learning to predict customer churn for a telecom company. It aims to identify customers at risk of leaving so that targeted retention strategies can be implemented.

## Key Features
- Comprehensive data preprocessing and feature engineering
- Advanced modeling with class imbalance handling
- Business-oriented evaluation metrics
- Optimized classification threshold for maximum ROI
- Feature importance analysis for actionable insights
- Interactive web application for making predictions

## Business Value
- The model achieves 83% accuracy in predicting customer churn
- When implemented with the optimal threshold, the model provides:
  - $498,780 value from retained customers
  - $8,900 cost of unnecessary retention offers
  - Net benefit of $377,320
  - ROI of 1,961%

## Key Insights
1. Month-to-Month contracts are the strongest predictor of churn
2. Number of referrals is strongly associated with customer loyalty
3. Monthly spend ratio (relative to tenure) significantly impacts churn probability

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
# Clone this repository
git clone https://github.com/yourusername/telecom-churn-prediction.git

# Go into the repository
cd telecom-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
