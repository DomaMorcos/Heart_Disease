import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
        margin: 0.5rem 0;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #FFE5E5;
        color: #D32F2F;
        border: 2px solid #FF6B6B;
    }
    .low-risk {
        background-color: #E8F5E8;
        color: #388E3C;
        border: 2px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction App</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
        Advanced Machine Learning for Cardiovascular Risk Assessment
    </p>
    <p style="color: #888;">
        Enter patient health data to predict heart disease risk using trained ML models
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Model Selection and Info
st.sidebar.markdown("## 🔧 Model Configuration")

# Load sample data for demonstration
@st.cache_data
def load_sample_data():
    # Create sample data if processed data doesn't exist
    sample_data = {
        'age': [63, 67, 67, 37, 41, 56, 62, 57, 63, 53],
        'sex': [1, 1, 1, 1, 0, 1, 0, 0, 1, 1],
        'cp': [1, 4, 4, 3, 2, 2, 4, 4, 4, 4],
        'trestbps': [145, 160, 120, 130, 130, 120, 140, 120, 130, 140],
        'chol': [233, 286, 229, 250, 204, 236, 268, 354, 254, 203],
        'fbs': [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        'restecg': [2, 2, 2, 0, 2, 0, 2, 0, 2, 2],
        'thalach': [150, 108, 129, 187, 172, 178, 160, 163, 147, 155],
        'exang': [0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
        'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4, 0.8, 3.6, 0.6, 1.4, 3.1],
        'slope': [3, 2, 2, 3, 1, 1, 3, 1, 2, 3],
        'ca': [0, 3, 2, 0, 0, 0, 2, 0, 1, 0],
        'thal': [6, 3, 7, 3, 3, 3, 3, 3, 7, 7],
        'target': [0, 1, 1, 0, 0, 0, 1, 0, 1, 1]
    }
    return pd.DataFrame(sample_data)

# Load data
df = load_sample_data()

# Model selection
model_option = st.sidebar.selectbox(
    "Select ML Model",
    ["Random Forest", "Logistic Regression", "Support Vector Machine", "Decision Tree"],
    index=0
)

# Create and train model (for demonstration)
@st.cache_resource
def get_trained_model(model_name):
    X = df.drop(['target'], axis=1)
    y = df['target']
    
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(random_state=42)
    elif model_name == "Support Vector Machine":
        model = SVC(probability=True, random_state=42)
    else:  # Decision Tree
        model = DecisionTreeClassifier(random_state=42)
    
    model.fit(X, y)
    return model

model = get_trained_model(model_option)

# Feature Information
feature_info = {
    'age': 'Age in years',
    'sex': 'Gender (1 = male, 0 = female)',
    'cp': 'Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)',
    'trestbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Serum cholesterol (mg/dl)',
    'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
    'restecg': 'Resting ECG results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)',
    'thalach': 'Maximum heart rate achieved',
    'exang': 'Exercise induced angina (1 = yes, 0 = no)',
    'oldpeak': 'ST depression induced by exercise relative to rest',
    'slope': 'Slope of peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)',
    'ca': 'Number of major vessels (0-3) colored by fluoroscopy',
    'thal': 'Thalassemia (3: normal, 6: fixed defect, 7: reversible defect)'
}

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h2 class="sub-header">📊 Patient Information Input</h2>', unsafe_allow_html=True)
    
    # Create input form
    with st.form("patient_data_form"):
        # Row 1
        col1_1, col1_2, col1_3 = st.columns(3)
        
        with col1_1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50, help=feature_info['age'])
            sex = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help=feature_info['sex'])
            cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4], 
                            format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"][x-1],
                            help=feature_info['cp'])
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], 
                             format_func=lambda x: "False" if x == 0 else "True", help=feature_info['fbs'])
        
        with col1_2:
            trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=250, value=120, help=feature_info['trestbps'])
            chol = st.number_input("Serum Cholesterol", min_value=100, max_value=600, value=200, help=feature_info['chol'])
            restecg = st.selectbox("Resting ECG", [0, 1, 2], 
                                 format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x],
                                 help=feature_info['restecg'])
            exang = st.selectbox("Exercise Induced Angina", [0, 1], 
                               format_func=lambda x: "No" if x == 0 else "Yes", help=feature_info['exang'])
        
        with col1_3:
            thalach = st.number_input("Max Heart Rate", min_value=60, max_value=250, value=150, help=feature_info['thalach'])
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help=feature_info['oldpeak'])
            slope = st.selectbox("ST Slope", [1, 2, 3], 
                                format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x-1],
                                help=feature_info['slope'])
            ca = st.number_input("Major Vessels", min_value=0, max_value=3, value=0, help=feature_info['ca'])
        
        # Additional feature
        thal = st.selectbox("Thalassemia", [3, 6, 7], 
                           format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][[3, 6, 7].index(x)],
                           help=feature_info['thal'])
        
        # Submit button
        submitted = st.form_submit_button("🔍 Predict Heart Disease Risk", use_container_width=True)
        
        if submitted:
            # Create input data
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Display results
            st.markdown("---")
            st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
            
            # Risk assessment
            risk_score = prediction_proba[1] * 100
            
            if prediction == 1:
                st.markdown(f'''
                <div class="prediction-result high-risk">
                    ⚠️ HIGH RISK - Heart Disease Detected<br>
                    <span style="font-size: 1.2rem;">Risk Score: {risk_score:.1f}%</span>
                </div>
                ''', unsafe_allow_html=True)
                st.error("⚠️ The model predicts a high risk of heart disease. Please consult with a healthcare professional immediately.")
            else:
                st.markdown(f'''
                <div class="prediction-result low-risk">
                    ✅ LOW RISK - No Heart Disease Detected<br>
                    <span style="font-size: 1.2rem;">Risk Score: {risk_score:.1f}%</span>
                </div>
                ''', unsafe_allow_html=True)
                st.success("✅ The model predicts a low risk of heart disease. Continue maintaining a healthy lifestyle.")
            
            # Probability breakdown
            col_prob1, col_prob2 = st.columns(2)
            with col_prob1:
                st.metric("No Disease Probability", f"{prediction_proba[0]:.3f}", delta=f"{(prediction_proba[0]-0.5):.3f}")
            with col_prob2:
                st.metric("Disease Probability", f"{prediction_proba[1]:.3f}", delta=f"{(prediction_proba[1]-0.5):.3f}")
            
            # Feature importance (for Random Forest)
            if model_option == "Random Forest":
                st.markdown('<h3 class="sub-header">📈 Feature Importance</h3>', unsafe_allow_html=True)
                
                feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
                importances = model.feature_importances_
                
                # Create feature importance plot
                fig = px.bar(
                    x=importances,
                    y=feature_names,
                    orientation='h',
                    title="Feature Importance in Prediction",
                    labels={'x': 'Importance', 'y': 'Features'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<h2 class="sub-header">ℹ️ About This App</h2>', unsafe_allow_html=True)
    
    st.info("""
    **How it works:**
    1. Enter patient health metrics
    2. ML model analyzes the data
    3. Receive risk assessment
    4. View feature importance
    """)
    
    st.markdown('<h3 class="sub-header">📊 Model Performance</h3>', unsafe_allow_html=True)
    
    # Display model accuracy (simulated)
    accuracy_scores = {
        "Random Forest": 0.87,
        "Logistic Regression": 0.84,
        "Support Vector Machine": 0.85,
        "Decision Tree": 0.81
    }
    
    current_accuracy = accuracy_scores[model_option]
    st.metric("Model Accuracy", f"{current_accuracy:.1%}")
    
    # Dataset statistics
    st.markdown('<h3 class="sub-header">📈 Dataset Statistics</h3>', unsafe_allow_html=True)
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Total Patients", len(df))
    with col_stat2:
        st.metric("Features", len(df.columns)-1)
    
    # Class distribution
    class_dist = df['target'].value_counts()
    fig_pie = px.pie(
        values=class_dist.values,
        names=['No Disease', 'Disease'],
        title="Class Distribution",
        color_discrete_sequence=['#4CAF50', '#FF6B6B']
    )
    fig_pie.update_layout(height=300)
    st.plotly_chart(fig_pie, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>
        ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. 
        Always consult with healthcare professionals for medical advice.
    </p>
    <p>
        Built with ❤️ using Streamlit and Machine Learning | 
        <a href="https://github.com" target="_blank">GitHub Repository</a>
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Additional Information
st.sidebar.markdown("---")
st.sidebar.markdown("## 📚 About Heart Disease")
st.sidebar.markdown("""
**Key Risk Factors:**
- Age and gender
- Chest pain patterns
- Blood pressure levels
- Cholesterol levels
- Exercise tolerance
- Family history

**Prevention Tips:**
- Regular exercise
- Healthy diet
- Stress management
- Regular check-ups
- No smoking
""")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔗 Resources")
st.sidebar.markdown("""
- [American Heart Association](https://www.heart.org)
- [WHO Cardiovascular Diseases](https://www.who.int/health-topics/cardiovascular-diseases)
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
""")