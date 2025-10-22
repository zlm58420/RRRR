import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set page config at the very beginning
st.set_page_config(page_title="Lung Nodule Risk Prediction", page_icon="🫁", layout="wide")

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .risk-moderate {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Check and load models function
@st.cache_resource
def load_models():
    """Safely load model files"""
    models = {}
    
    # Define possible model file paths
    model_files = {
        'small': ['S8mm_model.joblib', 'optimized_LR_small_nodules.pkl', 'best_model_small_nodules_external_LR.pkl'],
        'large': ['L8mm_model.joblib', 'optimized_LGB_large_nodules.pkl', 'best_model_large_nodules_external_LGB.pkl']
    }
    
    feature_files = {
        'small': ['S8mmfeatures.joblib'],
        'large': ['L8mmfeatures.joblib']
    }
    
    # Try to load small nodule model
    model_small = None
    for model_file in model_files['small']:
        try:
            if os.path.exists(model_file):
                model_small = joblib.load(model_file)
                st.success(f"✓ Loaded small nodule model: {model_file}")
                break
        except Exception as e:
            st.warning(f"Failed to load {model_file}: {str(e)}")
            continue
    
    # Try to load large nodule model
    model_large = None
    for model_file in model_files['large']:
        try:
            if os.path.exists(model_file):
                model_large = joblib.load(model_file)
                st.success(f"✓ Loaded large nodule model: {model_file}")
                break
        except Exception as e:
            st.warning(f"Failed to load {model_file}: {str(e)}")
            continue
    
    # Try to load features
    features_small = ['Age', 'Gender', 'Active or former smoker', 'Spiculated', 'Calcification', 
                     'Nodule diameter', 'CEA', 'SCC', 'Cyfra21_1', 'NSE']
    features_large = features_small.copy()
    
    for feature_file in feature_files['small']:
        try:
            if os.path.exists(feature_file):
                features_small = joblib.load(feature_file)
                st.success(f"✓ Loaded small nodule features: {feature_file}")
                break
        except Exception as e:
            st.warning(f"Failed to load feature file {feature_file}: {str(e)}")
            continue
    
    for feature_file in feature_files['large']:
        try:
            if os.path.exists(feature_file):
                features_large = joblib.load(feature_file)
                st.success(f"✓ Loaded large nodule features: {feature_file}")
                break
        except Exception as e:
            st.warning(f"Failed to load feature file {feature_file}: {str(e)}")
            continue
    
    return model_small, model_large, features_small, features_large

def get_user_input(features, nodule_diameter):
    """Get user input"""
    input_data = {}
    
    # Remove nodule diameter from features list since we get it separately
    features_without_diameter = [f for f in features if f != 'Nodule diameter']
    
    # Split features into two columns
    mid_point = len(features_without_diameter) // 2
    col1_features = features_without_diameter[:mid_point]
    col2_features = features_without_diameter[mid_point:]
    
    st.subheader("📋 Patient Clinical Features")
    
    col1, col2 = st.columns(2)
    
    # First column
    with col1:
        for feature in col1_features:
            if feature in ['Age', 'CEA', 'SCC', 'Cyfra21_1', 'NSE', 'ProGRP']:
                # Continuous variables
                input_data[feature] = st.number_input(
                    f"{feature}",
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    help=f"Enter {feature} value"
                )
            else:
                # Categorical variables
                if feature == 'Gender':
                    gender_option = st.selectbox(
                        "Gender",
                        ["Select", "Female", "Male"],
                        index=0
                    )
                    input_data[feature] = 0 if gender_option == "Female" else 1 if gender_option == "Male" else 0
                else:
                    option = st.selectbox(
                        f"{feature}",
                        ["Select", "No", "Yes"],
                        index=0
                    )
                    input_data[feature] = 0 if option == "No" else 1 if option == "Yes" else 0
    
    # Second column
    with col2:
        for feature in col2_features:
            if feature in ['Age', 'CEA', 'SCC', 'Cyfra21_1', 'NSE', 'ProGRP']:
                input_data[feature] = st.number_input(
                    f"{feature}",
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    help=f"Enter {feature} value",
                    key=f"{feature}_col2"
                )
            else:
                if feature == 'Gender':
                    gender_option = st.selectbox(
                        "Gender",
                        ["Select", "Female", "Male"],
                        index=0,
                        key="gender_col2"
                    )
                    input_data[feature] = 0 if gender_option == "Female" else 1 if gender_option == "Male" else 0
                else:
                    option = st.selectbox(
                        f"{feature}",
                        ["Select", "No", "Yes"],
                        index=0,
                        key=f"{feature}_col2"
                    )
                    input_data[feature] = 0 if option == "No" else 1 if option == "Yes" else 0
    
    # Add nodule diameter
    input_data['Nodule diameter'] = nodule_diameter
    
    return input_data

def display_prediction(malignancy_prob, input_data):
    """Display prediction results"""
    st.subheader("📊 Prediction Results")
    
    # Create metric cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Malignancy Probability",
            f"{malignancy_prob:.1%}",
            delta="High risk" if malignancy_prob > 0.5 else "Moderate risk" if malignancy_prob > 0.2 else "Low risk"
        )
    
    with col2:
        st.metric(
            "Risk Assessment",
            "High" if malignancy_prob >= 0.5 else "Moderate" if malignancy_prob >= 0.2 else "Low"
        )
    
    with col3:
        st.metric(
            "Recommended Action",
            "Immediate consultation" if malignancy_prob >= 0.5 else "Further evaluation" if malignancy_prob >= 0.2 else "Regular follow-up"
        )
    
    # Risk level display
    if malignancy_prob < 0.2:
        st.markdown('<div class="risk-low">🔵 Low Risk: Regular follow-up recommended</div>', unsafe_allow_html=True)
    elif malignancy_prob < 0.5:
        st.markdown('<div class="risk-moderate">🟡 Moderate Risk: Further evaluation suggested</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="risk-high">🔴 High Risk: Immediate clinical consultation advised</div>', unsafe_allow_html=True)
    
    # Probability bar
    st.subheader("Risk Probability Distribution")
    st.progress(float(malignancy_prob))
    st.caption(f"Current malignancy probability: {malignancy_prob:.1%}")

def main():
    """Main function"""
    st.markdown('<div class="main-header">🫁 Pulmonary Nodule Malignancy Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Show current working directory and file list (for debugging)
    if st.sidebar.checkbox("Show debug information"):
        st.sidebar.write("Current working directory:", os.getcwd())
        st.sidebar.write("Files in directory:", [f for f in os.listdir('.') if f.endswith(('.joblib', '.pkl'))])
    
    # Load models
    with st.spinner("Loading prediction models..."):
        model_small, model_large, features_small, features_large = load_models()
    
    # Check if models loaded successfully
    if model_small is None and model_large is None:
        st.error("""
        ❌ Unable to load prediction models. Please ensure the following files exist in the current directory:
        
        **Small Nodule Model Files (one of):**
        - S8mm_model.joblib
        - optimized_LR_small_nodules.pkl  
        - best_model_small_nodules_external_LR.pkl
        
        **Large Nodule Model Files (one of):**
        - L8mm_model.joblib
        - optimized_LGB_large_nodules.pkl
        - best_model_large_nodules_external_LGB.pkl
        
        **Feature Files (optional):**
        - S8mmfeatures.joblib
        - L8mmfeatures.joblib
        """)
        return
    
    # Sidebar - Nodule Information
    with st.sidebar:
        st.header("🔍 Nodule Information")
        nodule_diameter = st.slider(
            "Nodule Diameter (mm)",
            min_value=0.0,
            max_value=30.0,
            value=10.0,
            step=0.1,
            help="Measure the largest diameter of the pulmonary nodule"
        )
        
        # Model selection logic
        if nodule_diameter <= 8:
            if model_small is not None:
                st.success("**Small Nodule Model** (≤8mm)")
                current_model = model_small
                current_features = features_small
                model_type = "Small Nodule Model (Logistic Regression)"
            else:
                st.error("Small nodule model not available")
                return
        else:
            if model_large is not None:
                st.success("**Large Nodule Model** (>8mm)")
                current_model = model_large
                current_features = features_large
                model_type = "Large Nodule Model (LightGBM)"
            else:
                st.error("Large nodule model not available")
                return
        
        st.info(f"Using model: {model_type}")
        
        # Predict button
        predict_button = st.button(
            "🚀 Start Risk Assessment",
            type="primary",
            use_container_width=True
        )
    
    # Main content area
    if current_model is not None:
        # Get user input
        input_data = get_user_input(current_features, nodule_diameter)
        
        # Show input summary
        with st.expander("📋 Input Data Summary", expanded=True):
            st.json(input_data)
        
        # Execute prediction
        if predict_button:
            # Validate input completeness
            missing_fields = [k for k, v in input_data.items() if v == 0 and k not in ['Nodule diameter']]
            if missing_fields:
                st.warning(f"⚠️ Please complete the following fields: {', '.join(missing_fields)}")
            else:
                try:
                    with st.spinner("Performing risk assessment..."):
                        # Prepare input data
                        input_df = pd.DataFrame([input_data])
                        
                        # Ensure correct feature order
                        if hasattr(current_model, 'feature_names_in_'):
                            expected_features = current_model.feature_names_in_
                        else:
                            expected_features = current_features
                        
                        input_df = input_df[expected_features]
                        
                        # Execute prediction
                        prediction = current_model.predict_proba(input_df)
                        malignancy_prob = prediction[0][1]  # Get malignancy probability
                    
                    # Display results
                    display_prediction(malignancy_prob, input_data)
                    
                    # Show model information
                    with st.expander("ℹ️ Model Information"):
                        st.write(f"**Model Type:** {type(current_model).__name__}")
                        st.write(f"**Nodule Diameter:** {nodule_diameter} mm")
                        st.write(f"**Number of Features:** {len(expected_features)}")
                        st.write(f"**Feature List:** {', '.join(expected_features)}")
                        
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.info("Please check that all input fields are correctly filled")

if __name__ == "__main__":
    main()
