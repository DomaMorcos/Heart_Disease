# Heart Disease Prediction - Complete ML Pipeline

## 📊 Project Overview

This project implements a comprehensive machine learning pipeline for predicting heart disease using the UCI Heart Disease dataset. The pipeline includes data preprocessing, feature selection, dimensionality reduction (PCA), multiple classification models, unsupervised learning, hyperparameter tuning, and deployment via Streamlit.

## 🎯 Objectives

- ✅ Perform Data Preprocessing & Cleaning (handle missing values, encoding, scaling)
- ✅ Apply Dimensionality Reduction (PCA) to retain essential features
- ✅ Implement Feature Selection using statistical methods and ML-based techniques
- ✅ Train Supervised Learning Models (Logistic Regression, Decision Trees, Random Forest, SVM)
- ✅ Apply Unsupervised Learning (K-Means, Hierarchical Clustering) for pattern discovery  
- ✅ Optimize Models using Hyperparameter Tuning (GridSearchCV, RandomizedSearchCV)
- ✅ Deploy a Streamlit UI for real-time user interaction
- ✅ Host the application using Ngrok and upload to GitHub

## 🛠️ Tools & Technologies

- **Programming Language**: Python 3.8+
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly
- **ML Techniques**: PCA, RFE, Chi-Square Test, GridSearchCV, RandomizedSearchCV
- **Models**: Logistic Regression, Decision Trees, Random Forest, SVM, K-Means, Hierarchical Clustering
- **Deployment**: Streamlit, Ngrok
- **Version Control**: Git, GitHub

## 📁 Project Structure

```
Heart_Disease_Project/
│
├── data/
│   └── heart_disease.csv
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   └── 07_streamlit_deployment.ipynb
│
├── models/
│   └── final_model.pkl
│
├── ui/
│   └── app.py
│
├── deployment/
│   ├── deployment_report.json
│   └── QUICKSTART.md
│
├── results/
│   └── evaluation_metrics.txt
│
├── README.md
├── requirements.txt
└── .gitignore
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Heart_Disease_Project.git
cd Heart_Disease_Project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Jupyter Notebooks

Execute the notebooks in order:

```bash
jupyter notebook notebooks/01_data_preprocessing.ipynb
```

### 4. Launch Streamlit App

```bash
streamlit run ui/app.py
```
- Access locally at: http://localhost:8501

### 5. Deploy with Ngrok (Public Access)

```bash
# Terminal 1: Keep Streamlit running
streamlit run ui/app.py

# Terminal 2: Create public tunnel  
ngrok http 8501
```
- Use the provided ngrok URL for public access
- Share the link with others for remote testing

## 📊 Dataset Information

The UCI Heart Disease dataset contains:
- **303 instances** from Cleveland Clinic Foundation
- **14 attributes** used for prediction
- **Target variable**: Presence of heart disease (0-4, binary classification: 0 vs 1-4)

### Features:
1. **age**: Age in years
2. **sex**: Gender (1 = male, 0 = female)
3. **cp**: Chest pain type (1-4)
4. **trestbps**: Resting blood pressure
5. **chol**: Serum cholesterol
6. **fbs**: Fasting blood sugar > 120 mg/dl
7. **restecg**: Resting electrocardiographic results
8. **thalach**: Maximum heart rate achieved
9. **exang**: Exercise induced angina
10. **oldpeak**: ST depression induced by exercise
11. **slope**: Slope of peak exercise ST segment
12. **ca**: Number of major vessels colored by fluoroscopy
13. **thal**: Thalassemia type
14. **target**: Heart disease presence

## 🔍 Pipeline Steps

### 1. Data Preprocessing & Cleaning
- Load and combine multiple datasets
- Handle missing values using imputation
- Encode categorical variables
- Scale numerical features
- Exploratory Data Analysis (EDA)

### 2. Dimensionality Reduction (PCA)
- Apply PCA to reduce feature dimensionality
- Determine optimal number of components
- Visualize variance explained

### 3. Feature Selection
- Random Forest feature importance
- Recursive Feature Elimination (RFE)
- Chi-Square statistical test
- Select most relevant features

### 4. Supervised Learning
- **Models**: Logistic Regression, Decision Tree, Random Forest, SVM
- **Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Visualization**: ROC curves, confusion matrices

### 5. Unsupervised Learning
- **K-Means Clustering**: Elbow method for optimal K
- **Hierarchical Clustering**: Dendrogram analysis
- Compare clusters with actual labels

### 6. Hyperparameter Tuning
- GridSearchCV for exhaustive search
- RandomizedSearchCV for efficient search
- Cross-validation for robust evaluation

### 7. Model Deployment
- Save best model using joblib
- Create Streamlit web interface
- Deploy using Ngrok for public access

## 📈 Results

### Model Performance
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 85.2% | 0.83 | 0.87 | 0.85 | 0.91 |
| Random Forest | 87.5% | 0.86 | 0.89 | 0.87 | 0.93 |
| SVM | 84.8% | 0.82 | 0.88 | 0.85 | 0.90 |
| Decision Tree | 79.3% | 0.77 | 0.82 | 0.79 | 0.85 |

### Feature Importance
Top 5 most important features:
1. **thal** (Thalassemia type)
2. **ca** (Number of major vessels)
3. **oldpeak** (ST depression)
4. **thalach** (Maximum heart rate)
5. **cp** (Chest pain type)

## 🖥️ Web Interface

The Streamlit application provides:
- User-friendly input forms for health parameters
- Real-time heart disease prediction
- Probability scores and risk assessment
- Data visualization and insights
- Model performance metrics

## 🔗 Live Demo

Access the live application: [Ngrok Link] (Available when deployed)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: UCI Machine Learning Repository
- **Data Contributors**: 
  - Hungarian Institute of Cardiology, Budapest: Andras Janosi, M.D.
  - University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
  - University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
  - V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

## 📞 Contact

For questions or suggestions, please open an issue or contact [your-email@example.com].

---

⭐ **If you found this project helpful, please give it a star!** ⭐