# CrediSynth AI - MSME Lending Underwriter

An interactive web dashboard that uses a machine learning model to predict loan default risk for MSMEs (Micro, Small, and Medium Enterprises) and provides transparent, human-readable explanations for its decisions.

# Live Demo

You can access and interact with the live application here:

https://lendingunderwriter-credsynthai.streamlit.app/

# Overview

CrediSynth AI is an interactive tool designed to assist MSME lenders in making faster, more transparent credit decisions. It uses an XGBoost machine learning model to predict the probability of default based on key applicant data.

The most powerful feature is its explainability. Instead of just providing a "yes" or "no," the app uses a SHAP Waterfall Plot to show exactly which factors (like GST turnover, business vintage, or bank balance) contributed to the decision and by how much.

# Key Features
 - Instant Risk Prediction: Get an "Approve" or "Reject" decision in real-time.
 - Risk Score: See the precise probability of default (e.g., 15.2%).
 - AI-Powered Explanations: A dynamic SHAP plot shows which factors increased the risk (red bars) and which decreased it(blue bars).
 - Interactive "What-If" Analysis: Easily change applicant data in the sidebar to see how it affects the loan decision.

# Tech Stack
 * Python: Core programming language.
 * Streamlit: For building the interactive web dashboard.
 * Scikit-learn: For the machine learning pipeline (preprocessing, modeling).
 * XGBoost: The gradient boosting model used for prediction.
 * SHAP: For model explainability and generating decision plots.
 * Joblib: For saving and loading the trained model, explainer, and pipeline.
 * Pandas & Numpy: For data manipulation.

# Running Locally
To run this application on your local machine, follow these steps:

Clone the repository:
(Replace with your actual repository URL)

git clone https://github.com/lets-push-and-pray/lending_underwriter.git

cd lendingunderwriter-credsynthai


# Create and activate a virtual environment:

**Windows**
`python -m venv .venv`
`.venv\Scripts\activate`

**macOS / Linux**
`python3 -m venv .venv`
`source .venv/bin/activate`


**Install the dependencies:**
`pip install -r requirements.txt`


**Run the Streamlit app:**
`streamlit run app.py`


The app will open in your default web browser.

# Re-training the Model

The model and SHAP explainer are pre-trained and saved as .joblib files. If you want to re-train the model (e.g., after updating the data generation or training logic):

`python -m src.credi_synth_ai.train`


This will overwrite the model artifacts (model_pipeline.joblib, shap_explainer.joblib, feature_names.joblib) in the root directory.
