import numpy as np
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# --- Set page configuration ---
st.set_page_config(
    page_title="CrediSynth AI Underwriting",
    layout="wide"
)

# --- Load Model Artifacts ---
# Use a try-except block to handle file loading
try:
    pipeline = joblib.load("model_pipeline.joblib")
    explainer = joblib.load("shap_explainer.joblib")
    feature_names = joblib.load("feature_names.joblib")
except FileNotFoundError:
    st.error("Model files not found. Please run 'python -m src.credi_synth_ai.train' to train and save the models.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading model files: {e}")
    st.stop()

# Function to create the SHAP waterfall plot
def create_shap_plot(preprocessed_data, base_value, shap_values):
    """Creates a SHAP waterfall plot and returns the figure."""
    
    # Create the SHAP explanation object for a single prediction
    # We use shap_values[0] because we're explaining the first (and only) row
    # We also need to provide the base value and the processed data
    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=preprocessed_data,
        feature_names=feature_names
    )

    # Create the waterfall plot
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap_explanation, show=False)
    
    # Customize plot for better readability in Streamlit
    plt.title("Loan Decision Drivers", fontsize=16)
    plt.tight_layout()
    return fig

# --- Dashboard UI ---
st.title("🤖 CrediSynth AI - MSME Underwriting Assistant")
st.markdown("Enter the applicant's details in the sidebar to get an instant credit decision and explanation.")

# --- Sidebar for User Inputs ---
st.sidebar.header("Applicant Data")

# Create input fields
business_vintage_months = st.sidebar.number_input(
    "Business Vintage (Months)", min_value=1, max_value=240, value=45
)
avg_monthly_gst_turnover = st.sidebar.number_input(
    "Average Monthly GST Turnover (₹)", min_value=0, value=270000
)
avg_monthly_bank_balance = st.sidebar.number_input(
    "Average Monthly Bank Balance (₹)", min_value=0, value=129000
)
num_cheque_bounces_last_6mo = st.sidebar.number_input(
    "Cheque Bounces (Last 6 Months)", min_value=0, max_value=20, value=1
)
industry_type = st.sidebar.selectbox(
    "Industry Type",
    options=["Retail", "Service", "Manufacturing"]
)

# --- Main Page for Outputs ---
if st.sidebar.button("Analyze Application", type="primary", use_container_width=True):
    
    # 1. Collect inputs into a DataFrame
    # The column names MUST match those in the training data
    input_data = {
        'business_vintage_months': [business_vintage_months],
        'avg_monthly_gst_turnover': [avg_monthly_gst_turnover],
        'avg_monthly_bank_balance': [avg_monthly_bank_balance],
        'num_cheque_bounces_last_6mo': [num_cheque_bounces_last_6mo],
        'industry_type': [industry_type]
    }
    input_df = pd.DataFrame(input_data)

    # 2. Get Prediction & Probability
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0]
    
    # Get the risk score (probability of default)
    risk_score = probability[1] # [prob_0, prob_1]

    # --- Display Decision ---
    col1, col2 = st.columns(2)
    
    if prediction == 0:
        col1.success("## Decision: ✅ Approve Loan")
    else:
        col1.error("## Decision: ❌ Reject Application")
    
    col2.metric(
        label="Predicted Default Risk", 
        value=f"{risk_score:.2%}",
        help="This is the model's confidence that the applicant will default."
    )

    st.divider()

    # --- 3. Generate "Proper Reasons" with SHAP ---
    st.subheader("💡 Key Decision Factors")
    st.markdown("This chart shows how each factor *pushed* the decision from the 'base risk' to the 'final risk score'.")

    # Preprocess the single input row
    input_processed = pipeline.named_steps['preprocessor'].transform(input_df)

    # Get SHAP values for the single prediction
    # explainer(input_processed) returns a list of shap_values (one for each class)
    # For a binary classifier, we usually explain the "Default" (class 1)
    shap_values = explainer.shap_values(input_processed)
    
    # Get the explainer's base value (the average prediction)
    base_value = explainer.expected_value
    
    # Check if base_value is a list or array; if so, grab the second value.
    if isinstance(base_value, (list, np.ndarray)):
        shap_base_value = base_value[1]
    else:
        # If it's just a single number, use it as-is
        shap_base_value = base_value
    
    # Create and display the plot
    # We pass shap_values[1] (for class 1: "Default")
    # and input_processed (the data)
    fig = create_shap_plot(
        preprocessed_data=input_processed[0], 
        base_value=shap_base_value, 
        shap_values=shap_values[0]
    )
    st.pyplot(fig)