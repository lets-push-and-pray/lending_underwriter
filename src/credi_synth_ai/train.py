import pandas as pd
import xgboost as xgb
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from credi_synth_ai.data import generate_synthetic_data # <-- ADD THIS

# --- Define file paths for our saved models ---
PIPELINE_PATH = "model_pipeline.joblib"
EXPLAINER_PATH = "shap_explainer.joblib"
FEATURE_NAMES_PATH = "feature_names.joblib"


def train_model():
    """
    Full training pipeline: Loads data, trains model and SHAP,
    and saves the artifacts to disk.
    """
    print("--- Starting Model Training Pipeline ---")
    
    # 1. Load Data
    data = generate_synthetic_data(num_rows=2000)
    
    # Define features (X) and target (y)
    X = data.drop('loan_defaulted', axis=1)
    y = data['loan_defaulted']
    
    # Define categorical and numerical features
    categorical_features = ['industry_type']
    numerical_features = [
        'business_vintage_months', 
        'avg_monthly_gst_turnover',
        'avg_monthly_bank_balance',
        'num_cheque_bounces_last_6mo'
    ]

    # 2. Split data 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Calculate the base_score
    base_score_value = float(y_train.mean())
    
    # 4. Create a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    # 5. Create the full pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            random_state=42, 
            eval_metric='logloss',
            base_score=base_score_value
        ))
    ])
    
    # 6. Train the model
    model_pipeline.fit(X_train, y_train)

    # 7. Print evaluation report
    print("--- Model Evaluation Report ---")
    y_pred = model_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # --- 8. SHAP Explainer Setup (Re-added) ---
    print("--- Training SHAP Explainer ---")
    
    # Get the preprocessed training data
    # We use .transform() on X_train to get the data as the model sees it
    X_train_processed = model_pipeline.named_steps['preprocessor'].transform(X_train)
    
    # Get the feature names AFTER preprocessing
    all_feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()

    # Create a SHAP explainer
    explainer = shap.TreeExplainer(
        model_pipeline.named_steps['classifier'],
        X_train_processed
    )
    
    print("Model and SHAP Explainer successfully trained.")

    # --- 9. Save Artifacts to Disk ---
    joblib.dump(model_pipeline, PIPELINE_PATH)
    joblib.dump(explainer, EXPLAINER_PATH)
    joblib.dump(all_feature_names, FEATURE_NAMES_PATH) # Save feature names
    
    print(f"Model pipeline saved to {PIPELINE_PATH}")
    print(f"SHAP explainer saved to {EXPLAINER_PATH}")
    print(f"Feature names saved to {FEATURE_NAMES_PATH}")


# This block allows us to run "python -m src.credi_synth_ai.train"
if __name__ == "__main__":
    train_model()