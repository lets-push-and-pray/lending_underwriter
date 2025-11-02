import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def train_model(data: pd.DataFrame):
    """
    Preprocesses data and trains an XGBoost model.
    Returns the fitted pipeline and the SHAP explainer.
    """
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
    
    # Create a preprocessor
    # OneHotEncoder for categorical data, 'passthrough' for numerical
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )
    
    # Create the full pipeline with the XGBoost classifier
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            random_state=42, 
            use_label_encoder=False, 
            eval_metric='logloss'
        ))
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the model
    model_pipeline.fit(X_train, y_train)
    
    # Print evaluation report
    print("Model Evaluation Report:")
    y_pred = model_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # --- SHAP Explainer Setup ---
    # Get the preprocessed training data
    X_train_processed = model_pipeline.named_steps['preprocessor'].transform(X_train)
    
    # Get the feature names after OneHotEncoding
    cat_features_out = model_pipeline.named_steps['preprocessor'] \
                        .named_transformers_['cat'] \
                        .get_feature_names_out(categorical_features)
    
    # Combine all feature names in the correct order
    all_feature_names = numerical_features + list(cat_features_out)
    
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(
        model_pipeline.named_steps['classifier'],
        X_train_processed,
        feature_names=all_feature_names
    )
    
    print("Model and SHAP Explainer successfully trained.")
    
    return model_pipeline, explainer, X_test, y_test