
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="User Retention & Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------------
# Load Model Files
# ----------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("best_churn_model.pkl")
    threshold = joblib.load("optimal_threshold.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, threshold, preprocessor

try:
    model, threshold, preprocessor = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")

# ----------------------------------------
# Header
# ----------------------------------------
st.title("📊 User Retention & Churn Predictor")
st.markdown("""
**Built by Vijayalakshmi Veeraiyan**  
*Gradient Boosting Model | ROC AUC: 0.8931 | 
Accuracy: 83.25% | Recall: 80%*
""")
st.markdown("---")

# ----------------------------------------
# Sidebar — About
# ----------------------------------------
with st.sidebar:
    st.header("ℹ️ About This App")
    st.markdown("""
    This app predicts whether a telecom 
    customer is likely to churn — enabling 
    targeted retention strategies.
    
    **Model:** Gradient Boosting  
    **Threshold:** 0.3049  
    **Dataset:** 7,043 customers  
    
    **Business Assumptions:**
    - CLV: $1,680
    - Retention offer cost: $50
    
    **GitHub:**  
    [View Project](https://github.com/Viji-Veer/MLprojects)
    
    **LinkedIn:**  
    [Connect with Vijayalakshmi Veeraiyan](https://www.linkedin.com/in/vijayalakshmi-veeraiyan)
    """)
    
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("""
    1. Fill in customer details
    2. Click Predict
    3. See churn risk and recommendation
    """)

# ----------------------------------------
# Input Form
# ----------------------------------------
st.header("👤 Enter Customer Details")
st.markdown("Fill in the customer information below:")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    age = st.number_input(
        "Age", 
        min_value=18, max_value=100, value=35
    )
    gender = st.selectbox(
        "Gender", 
        ["Male", "Female"]
    )
    married = st.selectbox(
        "Married", 
        ["Yes", "No"]
    )
    number_of_dependents = st.number_input(
        "Number of Dependents", 
        min_value=0, max_value=10, value=0
    )
    number_of_referrals = st.number_input(
        "Number of Referrals", 
        min_value=0, max_value=20, value=0
    )

with col2:
    st.subheader("Contract & Billing")
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-Month", "One Year", "Two Year"]
    )
    tenure_in_months = st.number_input(
        "Tenure (Months)", 
        min_value=1, max_value=72, value=12
    )
    monthly_charge = st.number_input(
        "Monthly Charge ($)", 
        min_value=0.0, max_value=200.0, 
        value=65.0, step=0.1
    )
    total_charges = st.number_input(
        "Total Charges ($)", 
        min_value=0.0, max_value=10000.0, 
        value=780.0, step=0.1
    )
    paperless_billing = st.selectbox(
        "Paperless Billing", 
        ["Yes", "No"]
    )
    payment_method = st.selectbox(
        "Payment Method",
        ["Credit Card", "Bank Withdrawal", 
         "Mailed Check"]
    )

with col3:
    st.subheader("Services")
    phone_service = st.selectbox(
        "Phone Service", 
        ["Yes", "No"]
    )
    internet_service = st.selectbox(
        "Internet Service", 
        ["Yes", "No"]
    )
    internet_type = st.selectbox(
        "Internet Type",
        ["Fiber Optic", "Cable", "DSL", "None"]
    )
    online_security = st.selectbox(
        "Online Security", 
        ["Yes", "No"]
    )
    online_backup = st.selectbox(
        "Online Backup", 
        ["Yes", "No"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV", 
        ["Yes", "No"]
    )
    unlimited_data = st.selectbox(
        "Unlimited Data", 
        ["Yes", "No"]
    )

# Additional fields in expander
with st.expander("📋 Additional Details (Optional)"):
    col4, col5 = st.columns(2)
    with col4:
        multiple_lines = st.selectbox(
            "Multiple Lines", 
            ["Yes", "No"]
        )
        device_protection = st.selectbox(
            "Device Protection Plan", 
            ["Yes", "No"]
        )
        premium_tech_support = st.selectbox(
            "Premium Tech Support", 
            ["Yes", "No"]
        )
    with col5:
        streaming_movies = st.selectbox(
            "Streaming Movies", 
            ["Yes", "No"]
        )
        streaming_music = st.selectbox(
            "Streaming Music", 
            ["Yes", "No"]
        )
        avg_monthly_gb = st.number_input(
            "Avg Monthly GB Download", 
            min_value=0.0, max_value=100.0, 
            value=10.0
        )
        avg_long_distance = st.number_input(
            "Avg Monthly Long Distance ($)", 
            min_value=0.0, max_value=100.0, 
            value=10.0
        )

# ----------------------------------------
# Prediction Button
# ----------------------------------------
st.markdown("---")
predict_button = st.button(
    "🔮 Predict Churn Risk", 
    type="primary",
    use_container_width=True
)

# ----------------------------------------
# Prediction Logic
# ----------------------------------------
if predict_button and model_loaded:

    # Build input dataframe matching 
    # training data structure
    input_data = {
        'Gender': gender,
        'Age': age,
        'Married': married,
        'Number of Dependents': number_of_dependents,
        'Number of Referrals': number_of_referrals,
        'Tenure in Months': tenure_in_months,
        'Offer': 'None',
        'Phone Service': phone_service,
        'Avg Monthly Long Distance Charges': avg_long_distance,
        'Multiple Lines': multiple_lines,
        'Internet Service': internet_service,
        'Internet Type': internet_type,
        'Avg Monthly GB Download': avg_monthly_gb,
        'Online Security': online_security,
        'Online Backup': online_backup,
        'Device Protection Plan': device_protection,
        'Premium Tech Support': premium_tech_support,
        'Streaming TV': streaming_tv,
        'Streaming Movies': streaming_movies,
        'Streaming Music': streaming_music,
        'Unlimited Data': unlimited_data,
        'Contract': contract,
        'Paperless Billing': paperless_billing,
        'Payment Method': payment_method,
        'Monthly Charge': monthly_charge,
        'Total Charges': total_charges,
        'Total Refunds': 0.0,
        'Total Extra Data Charges': 0,
        'Total Long Distance Charges': avg_long_distance * tenure_in_months,
        'Total Revenue': total_charges,
        'Zip Code': '90001',
        'Latitude': 34.05,
        'Longitude': -118.24,
        'City': 'Los Angeles',
        'Tenure_Years': tenure_in_months / 12,
        'Age_Group': pd.cut(
            [age], 
            bins=[0, 30, 45, 60, 100],
            labels=['Under 30', '30-45', 
                   '46-60', 'Over 60']
        )[0],
        'Premium_Services_Count': sum([
            online_security == 'Yes',
            online_backup == 'Yes',
            device_protection == 'Yes',
            premium_tech_support == 'Yes',
            streaming_tv == 'Yes',
            streaming_movies == 'Yes',
            streaming_music == 'Yes'
        ]),
        'High_Data_User': 1 if avg_monthly_gb > 50 else 0,
        'Monthly_Spend_Ratio': monthly_charge / max(tenure_in_months, 1)
    }

    # Convert to dataframe
    input_df = pd.DataFrame([input_data])

    try:
        # Preprocess
        input_processed = preprocessor.transform(input_df)

        # Predict probability
        churn_probability = model.predict_proba(
            input_processed
        )[0][1]

        # Apply optimal threshold
        is_churn = churn_probability >= threshold

        # Business calculations
        clv = 1680
        retention_cost = 50
        retention_value = clv - retention_cost

        # ----------------------------------------
        # Display Results
        # ----------------------------------------
        st.markdown("---")
        st.header("🎯 Prediction Results")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            if is_churn:
                st.error("🔴 HIGH CHURN RISK")
                st.metric(
                    "Churn Probability",
                    f"{churn_probability:.1%}"
                )
            else:
                st.success("🟢 LOW CHURN RISK")
                st.metric(
                    "Churn Probability",
                    f"{churn_probability:.1%}"
                )

        with result_col2:
            st.metric(
                "Model Threshold",
                f"{threshold:.4f}"
            )
            st.metric(
                "Customer Lifetime Value",
                f"${clv:,}"
            )

        with result_col3:
            if is_churn:
                st.metric(
                    "Recommended Action",
                    "Send Retention Offer"
                )
                st.metric(
                    "Value at Stake",
                    f"${retention_value:,}"
                )
            else:
                st.metric(
                    "Recommended Action",
                    "No Action Needed"
                )
                st.metric(
                    "Value at Stake",
                    "$0"
                )

        # Recommendation box
        st.markdown("---")
        if is_churn:
            st.warning(f"""
            ### ⚠️ Retention Recommendation
            
            This customer has a **{churn_probability:.1%} probability** 
            of churning — above the optimal threshold of 
            {threshold:.4f}.
            
            **Recommended action:** Send a targeted retention 
            offer immediately.
            
            **Cost of offer:** ${retention_cost}  
            **Value preserved if retained:** ${retention_value:,}  
            **ROI of intervention:** {retention_value/retention_cost:.0f}x
            
            **Key risk factors for this customer:**
            - Contract type: {contract}
            - Tenure: {tenure_in_months} months
            - Monthly spend ratio: 
              ${monthly_charge/max(tenure_in_months,1):.2f}
            - Number of referrals: {number_of_referrals}
            """)
        else:
            st.success(f"""
            ### ✅ Customer Appears Loyal
            
            This customer has a **{churn_probability:.1%} probability** 
            of churning — below the optimal threshold of 
            {threshold:.4f}.
            
            **Recommended action:** No immediate intervention needed.
            
            Continue monitoring and maintain service quality.
            """)

        # Probability gauge
        st.markdown("---")
        st.subheader("📈 Churn Probability Gauge")

        gauge_col1, gauge_col2 = st.columns([3, 1])
        with gauge_col1:
            st.progress(float(churn_probability))
        with gauge_col2:
            st.write(f"**{churn_probability:.1%}**")

        st.caption(f"""
        Threshold: {threshold:.4f} | 
        Below threshold = Low risk | 
        Above threshold = High risk
        """)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info("""
        Make sure all model files are in 
        the same folder as app.py:
        - best_churn_model.pkl
        - optimal_threshold.pkl  
        - preprocessor.pkl
        """)

elif predict_button and not model_loaded:
    st.error("""
    Model files not found. Please ensure 
    these files are in the same folder as app.py:
    - best_churn_model.pkl
    - optimal_threshold.pkl
    - preprocessor.pkl
    """)

# ----------------------------------------
# Footer
# ----------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: grey;'>
Built by Vijayalakshmi Veeraiyan  | 
Vantaa, Finland | 
<a href='https://github.com/Viji-Veer/MLprojects'>
GitHub</a> | 
<a href='https://www.linkedin.com/in/vijayalakshmi-veeraiyan'>
LinkedIn</a>
</div>
""", unsafe_allow_html=True)
