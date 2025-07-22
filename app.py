import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Define the directory to save the models and preprocessors
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

# Save the trained models and preprocessors
joblib.dump(model, os.path.join(model_dir, 'decision_tree_model.pkl'))
joblib.dump(svm_model, os.path.join(model_dir, 'svm_model.pkl'))
joblib.dump(rf_model, os.path.join(model_dir, 'random_forest_model.pkl'))
joblib.dump(lr_model, os.path.join(model_dir, 'logistic_regression_model.pkl'))
joblib.dump(grid.best_estimator_, os.path.join(model_dir, 'svm_gridsearch_model.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))

# Streamlit App Code (to be saved as app.py)
streamlit_code = """
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load the trained models and preprocessors
model_dir = "saved_models"
try:
    model = joblib.load(os.path.join(model_dir, 'decision_tree_model.pkl'))
    svm_model = joblib.load(os.path.join(model_dir, 'svm_model.pkl'))
    rf_model = joblib.load(os.path.join(model_dir, 'random_forest_model.pkl'))
    lr_model = joblib.load(os.path.join(model_dir, 'logistic_regression_model.pkl'))
    svm_gridsearch_model = joblib.load(os.path.join(model_dir, 'svm_gridsearch_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'saved_models' directory with model files exists.")
    st.stop()


st.title("Personality Type Predictor")
st.write("Predict personality type using different machine learning models.")

# Assuming the original column names are available or can be inferred
# from the loaded scaler or a sample of the original data if needed.
# For this example, I'll use the columns from the X_train DataFrame
# which should be available if you ran the training code before generating this app.py
# In a standalone app.py, you might need to load a sample of your data
# or have a predefined list of column names.
# Let's assume X is available from the notebook context for generating this code
# In a real app.py, you'd need the list of feature names.

# Create input widgets for each feature
input_features = {}
# Replace X.columns with the actual list of your feature names if running standalone
feature_names = ['social_energy', 'alone_time_preference', 'talkativeness', 'deep_reflection', 'group_comfort', 'party_liking', 'listening_skill', 'empathy', 'creativity', 'organization', 'leadership', 'risk_taking', 'public_speaking_comfort', 'curiosity', 'routine_preference', 'excitement_seeking', 'friendliness', 'emotional_stability', 'planning', 'spontaneity', 'adventurousness', 'reading_habit', 'sports_interest', 'online_social_usage', 'travel_desire', 'gadget_usage', 'work_style_collaborative', 'decision_speed', 'stress_handling'] # Example feature names
for col in feature_names:
    input_features[col] = st.slider(f"Enter value for {col}", 0.0, 10.0, 5.0) # Assuming original values were 0-10

# Convert input features to a DataFrame
input_df = pd.DataFrame([input_features])

# Scale the input features using the fitted scaler
input_df[feature_names] = scaler.transform(input_df[feature_names])


# Model selection
model_name = st.selectbox("Select Model", ['Decision Tree', 'SVM', 'Random Forest', 'Logistic Regression', 'SVM (GridSearchCV)'])

# Make prediction
if st.button("Predict"):
    # Select the appropriate model
    if model_name == 'Decision Tree':
        model_to_use = model
    elif model_name == 'SVM':
        model_to_use = svm_model
    elif model_name == 'Random Forest':
        model_to_use = rf_model
    elif model_name == 'Logistic Regression':
        model_to_use = lr_model
    elif model_name == 'SVM (GridSearchCV)':
        model_to_use = svm_gridsearch_model # Use the best estimator from GridSearchCV

    # Get the prediction (which will be a numerical label)
    predicted_label = model_to_use.predict(input_df)[0]

    # Inverse transform the predicted label to get the original personality type string
    predicted_personality_type = label_encoder.inverse_transform([predicted_label])[0]

    st.success(f"Predicted Personality Type: {predicted_personality_type}")
"""

# Save the Streamlit code to a file named app.py
with open("app.py", "w") as f:
    f.write(streamlit_code)

print("app.py created successfully. Save the models in the 'saved_models' directory alongside app.py to run it in Visual Studio Code.")