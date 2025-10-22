import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set page config at the very beginning
st.set_page_config(page_title="Lung Nodule Risk Prediction", page_icon="🫁", layout="wide")

# Try to import LightGBM, handle if not available
try:
    import lightgbm
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    st.warning("⚠️ LightGBM is not available. Large nodule model will not work properly.")

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
    .model-info {
        background-color: #e9ecef;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Define fixed feature set based on your optimization results
FEATURE_SET = ['Gender', 'Spiculated', 'Age', 'Nodule diameter', 'CEA', 'SCC', 'Cyfra21_1', 'NSE']

# Check and load models function
@st.cache_resource
def load_models():
    """Safely load model files"""
    
    # Define model file paths
    model_files = {
        'small': 'optimized_LR_small_nodules.pkl',
        'large': 'optimized_LGB_large_nodules.pkl'
    }
    
    # Try to load small nodule model
    model_small = None
    if os.path.exists(model_files['small']):
        try:
            model_small = joblib.load(model_files['small'])
            st.success(f"✓ Loaded small nodule model: {model_files['small']}")
        except Exception as e:
            st.error(f"Failed to load small nodule model: {str(e)}")
    else:
        st.error(f"Small nodule model file not found: {model_files['small']}")
    
    # Try to load large nodule model
    model_large = None
    if os.path.exists(model_files['large']):
        try:
            if not LIGHTGBM_AVAILABLE:
                st.error(f"Cannot load {model_files['large']}: LightGBM not installed")
            else:
                model_large = joblib.load(model_files['large'])
                st.success(f"✓ Loaded large nodule model: {model_files['large']}")
        except Exception as e:
            error_msg = str(e)
            if "lightgbm" in error_msg.lower():
                st.error(f"Cannot load {model_files['large']}: LightGBM not installed")
            else:
                st.error(f"Failed to load large nodule model: {error_msg}")
    else:
        st.error(f"Large nodule model file not found: {model_files['large']}")
    
    return model_small, model_large

def get_user_input(nodule_diameter):
    """Get user input for the fixed feature set"""
    input_data = {}
    
    st.subheader("📋 Patient Clinical Features")
    st.info(f"Using optimized feature set: {', '.join(FEATURE_SET)}")
    
    # Split features into two columns
    mid_point = len(FEATURE_SET) // 2
    col1_features = FEATURE_SET[:mid_point]
    col2_features = FEATURE_SET[mid_point:]
    
    col1, col2 = st.columns(2)
    
    # First column
    with col1:
        for feature in col1_features:
            if feature == 'Nodule diameter':
                # Already set from sidebar, just display
                st.write(f"**Nodule Diameter:** {nodule_diameter} mm")
                input_data[feature] = nodule_diameter
            elif feature in ['Age', 'CEA', 'SCC', 'Cyfra21_1', 'NSE']:
                # Continuous variables
                input_data[feature] = st.number_input(
                    f"{feature}",
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    help=f"Enter {feature} value"
                )
            elif feature == 'Gender':
                gender_option = st.selectbox(
                    "Gender",
                    ["Select", "Female", "Male"],
                    index=0
                )
                input_data[feature] = 0 if gender_option == "Female" else 1 if gender_option == "Male" else 0
            elif feature == 'Spiculated':
                option = st.selectbox(
                    f"{feature}",
                    ["Select", "No", "Yes"],
                    index=0
                )
                input_data[feature] = 0 if option == "No" else 1 if option == "Yes" else 0
    
    # Second column
    with col2:
        for feature in col2_features:
            if feature == 'Nodule diameter':
                # Already set from sidebar, just display
                st.write(f"**Nodule Diameter:** {nodule_diameter} mm")
                input_data[feature] = nodule_diameter
            elif feature in ['Age', 'CEA', 'SCC', 'Cyfra21_1', 'NSE']:
                input_data[feature] = st.number_input(
                    f"{feature}",
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    help=f"Enter {feature} value",
                    key=f"{feature}_col2"
                )
            elif feature == 'Gender':
                gender_option = st.selectbox(
                    "Gender",
                    ["Select", "Female", "Male"],
                    index=0,
                    key="gender_col2"
                )
                input_data[feature] = 0 if gender_option == "Female" else 1 if gender_option == "Male" else 0
            elif feature == 'Spiculated':
                option = st.selectbox(
                    f"{feature}",
                    ["Select", "No", "Yes"],
                    index=0,
                    key=f"{feature}_col2"
                )
                input_data[feature] = 0 if option == "No" else 1 if option == "Yes" else 0
    
    return input_data

def display_prediction(malignancy_prob, input_data, model_type):
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
    
    # Model performance info
    st.markdown("---")
    st.subheader("ℹ️ Model Information")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown('<div class="model-info">', unsafe_allow_html=True)
        st.write(f"**Model Type:** {model_type}")
        st.write(f"**Validation AUC:** 0.860 (Small) / 0.853 (Large)")
        st.write(f"**Features Used:** {len(FEATURE_SET)}")
        st.write("</div>", unsafe_allow_html=True)
    
    with col_info2:
        st.markdown('<div class="model-info">', unsafe_allow_html=True)
        st.write("**Optimized Feature Set:**")
        for feature in FEATURE_SET:
            st.write(f"- {feature}")
        st.write("</div>", unsafe_allow_html=True)

def main():
    """Main function"""
    st.markdown('<div class="main-header">🫁 Pulmonary Nodule Malignancy Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Show dependency status
    if not LIGHTGBM_AVAILABLE:
        st.error("""
        ❌ LightGBM is not installed. Large nodule predictions will not work.
        
        **To fix this, run:**  
        `pip install lightgbm`
        """)
    
    # Load models
    with st.spinner("Loading prediction models..."):
        model_small, model_large = load_models()
    
    # Check if models loaded successfully
    if model_small is None:
        st.error("""
        ❌ Unable to load small nodule model. Please ensure the following file exists:
        - optimized_LR_small_nodules.pkl
        """)
        return
    
    if model_large is None and LIGHTGBM_AVAILABLE:
        st.error("""
        ❌ Unable to load large nodule model. Please ensure the following file exists:
        - optimized_LGB_large_nodules.pkl
        """)
    
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
                model_type = "Optimized Logistic Regression (Small Nodules)"
                expected_auc = "0.860"
            else:
                st.error("Small nodule model not available")
                return
        else:
            if model_large is not None and LIGHTGBM_AVAILABLE:
                st.success("**Large Nodule Model** (>8mm)")
                current_model = model_large
                model_type = "Optimized LightGBM (Large Nodules)"
                expected_auc = "0.853"
            else:
                st.error("Large nodule model not available")
                st.info("Please install LightGBM to enable large nodule predictions")
                return
        
        st.info(f"Using: {model_type}")
        st.info(f"Expected AUC: {expected_auc}")
        
        # Show feature set in sidebar
        with st.expander("Feature Set (8 features)"):
            for feature in FEATURE_SET:
                st.write(f"- {feature}")
        
        # Predict button
        predict_button = st.button(
            "🚀 Start Risk Assessment",
            type="primary",
            use_container_width=True
        )
    
    # Main content area
    if current_model is not None:
        # Get user input
        input_data = get_user_input(nodule_diameter)
        
        # Show input summary
        with st.expander("📋 Input Data Summary", expanded=True):
            # Create a nicer display of input data
            col1, col2 = st.columns(2)
            with col1:
                for i, (key, value) in enumerate(input_data.items()):
                    if i < len(input_data) // 2:
                        st.write(f"**{key}:** {value}")
            with col2:
                for i, (key, value) in enumerate(input_data.items()):
                    if i >= len(input_data) // 2:
                        st.write(f"**{key}:** {value}")
        
        # Execute prediction
        if predict_button:
            # Validate input completeness
            missing_fields = [k for k, v in input_data.items() if v == 0 and k not in ['Nodule diameter']]
            if missing_fields:
                st.warning(f"⚠️ Please complete the following fields: {', '.join(missing_fields)}")
            else:
                try:
                    with st.spinner("Performing risk assessment..."):
                        # Prepare input data - ensure correct order
                        input_df = pd.DataFrame([input_data])[FEATURE_SET]
                        
                        # Execute prediction
                        prediction = current_model.predict_proba(input_df)
                        malignancy_prob = prediction[0][1]  # Get malignancy probability
                    
                    # Display results
                    display_prediction(malignancy_prob, input_data, model_type)
                        
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.info("Please check that all input fields are correctly filled")

if __name__ == "__main__":
    main()
