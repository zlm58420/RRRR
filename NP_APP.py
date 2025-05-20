import streamlit as st

# 将 set_page_config 移到最前面
st.set_page_config(page_title="Lung Nodule Risk Prediction", page_icon="🫁", layout="wide")

# 后续的导入和代码
import numpy as np
import packaging.version
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Custom CSS for styling
st.markdown("""
<style>
.main-container {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
}
.stNumberInput > div > div > input {
    background-color: white;
    border: 1px solid #4a84c4;
    border-radius: 5px;
}
.stSelectbox > div > div > select {
    background-color: white;
    border: 1px solid #4a84c4;
    border-radius: 5px;
}
.stButton > button {
    background-color: #4a84c4;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
}
.stMetric > div {
    background-color: white;
    border-radius: 10px;
    padding: 10px;
    text-align: center;
}
.stMetric > div > div {
    color: #4a84c4;
}
</style>
""", unsafe_allow_html=True)

# Load models and features
model_8mm = joblib.load('GBC_8mm_model.joblib')
model_30mm = joblib.load('GBC_30mm_model.joblib')
features_8mm = joblib.load('GBC_8mm_features.joblib')
features_30mm = joblib.load('GBC_30mm_features.joblib')


# 3. 创建输入表单
def get_user_input(features, nodule_diameter):
    input_data = {}
    
    # 将特征平均分配到两列
    mid = len(features) // 2
    col1_features = features[:mid]
    col2_features = features[mid:]
    
    # 创建两列
    col1, col2 = st.columns(2)
    
    # 第一列处理
    with col1:
        for feature in col1_features:
            if feature == 'Nodule diameter':
                st.write(f"Nodule Diameter (mm): {nodule_diameter}")
                input_data[feature] = nodule_diameter
            elif feature in ['Age', 'CEA', 'SCC', 'Cyfra21_1', 'NSE', 'ProGRP']:
                input_data[feature] = st.number_input(
                    f"Enter {feature}", 
                    min_value=0.0, 
                    step=0.1, 
                    key=f"{feature}_col1"
                )
            else:
                input_data[feature] = st.selectbox(
                    f"Select {feature}", 
                    [0, 1], 
                    format_func=lambda x, f=feature: 
                        "Female" if f == 'Gender' and x == 0 else 
                        "Male" if f == 'Gender' and x == 1 else 
                        "No" if x == 0 else "Yes",
                    key=f"{feature}_col1"
                )
    
    # 第二列处理
    with col2:
        for feature in col2_features:
            if feature == 'Nodule diameter':
                st.write(f"Nodule Diameter (mm): {nodule_diameter}")
                input_data[feature] = nodule_diameter
            elif feature in ['Age', 'CEA', 'SCC', 'Cyfra21_1', 'NSE', 'ProGRP']:
                input_data[feature] = st.number_input(
                    f"Enter {feature}", 
                    min_value=0.0, 
                    step=0.1, 
                    key=f"{feature}_col2"
                )
            else:
                input_data[feature] = st.selectbox(
                    f"Select {feature}", 
                    [0, 1], 
                    format_func=lambda x, f=feature: 
                        "Female" if f == 'Gender' and x == 0 else 
                        "Male" if f == 'Gender' and x == 1 else 
                        "No" if x == 0 else "Yes",
                    key=f"{feature}_col2"
                )
    
    return input_data

def predict_and_explain(input_data, model, features):
    # 确保按照原始特征顺序创建 DataFrame
    input_df = pd.DataFrame([{feature: input_data[feature] for feature in features}])
    
    # 后续代码保持不变
    prediction = model.predict_proba(input_df)
    malignancy_prob = prediction[0][1]

    # SHAP Explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    # 处理 SHAP 值的索引问题
    shap_plot_values = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    # Results Display
    result_col1, result_col2 = st.columns([1, 1])
    
    with result_col1:
        st.markdown("### Prediction Results")
        st.metric("Malignancy Risk", f"{malignancy_prob:.2%}")
    
    with result_col2:
        st.markdown("### Risk Interpretation")
        if malignancy_prob < 0.2:
            st.success("Low Risk: Close monitoring recommended")
        elif malignancy_prob < 0.5:
            st.warning("Moderate Risk: Further investigation suggested")
        else:
            st.error("High Risk: Immediate clinical consultation advised")
    
    # SHAP Visualizations
    st.markdown("### Feature Impact Analysis")
    col_shap1, col_shap2 = st.columns(2)
    
    with col_shap1:
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_plot_values, input_df, plot_type="bar", show=False)
        st.pyplot(fig)
    
    with col_shap2:
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_plot_values, input_df, show=False)
        st.pyplot(fig)

def main():
    st.title("🫁 TEB Lung Nodule Malignancy Risk Predictor")
    
    with st.sidebar:
        st.header("Model Selection")
        nodule_diameter = st.number_input("Nodule Diameter (mm)", min_value=0.0, max_value=50.0, step=0.1)
        
        if nodule_diameter <= 8:
            st.info("8mm Model Selected")
            features = features_8mm
            model = model_8mm
            print("8mm Model Features:", features)  # 打印特征
        elif nodule_diameter <= 30:
            st.info("30mm Model Selected")
            features = features_30mm
            model = model_30mm
            print("30mm Model Features:", features)  # 打印特征
        else:
            st.error("Nodule diameter out of predictive range")
            return
    
    input_data = get_user_input(features, nodule_diameter)
    
    if st.sidebar.button("Predict Risk"):
        predict_and_explain(input_data, model, features)

if __name__ == "__main__":
    main()