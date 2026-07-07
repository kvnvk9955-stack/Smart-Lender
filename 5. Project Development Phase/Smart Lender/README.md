# 🏦 Smart Lender — AI-Powered Loan Eligibility Predictor


Smart Lender is an AI-powered web application that predicts loan approval eligibility using Machine Learning. The application analyzes applicant information such as income, credit history, education, employment status, and loan details to provide instant loan approval predictions along with confidence scores.

Designed with a modern user interface and powered by an optimized XGBoost model, Smart Lender demonstrates the complete lifecycle of a Machine Learning project—from data preprocessing and model training to deployment as a production-ready Flask web application.

---

## 🌐 Live Demo

🔗 **Application:** https://kvnvk.pythonanywhere.com

---

## 📖 Project Overview

Financial institutions receive thousands of loan applications every day. Manual verification is often time-consuming and prone to inconsistencies.

Smart Lender simplifies this process by leveraging Machine Learning to predict whether a loan application is likely to be approved. The application processes applicant information, performs necessary preprocessing, feeds the data into a trained XGBoost classifier, and instantly returns the prediction with a confidence score.

The project demonstrates an end-to-end Machine Learning workflow, including:

* Data Collection
* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Data Preprocessing
* Model Training
* Hyperparameter Tuning
* Model Evaluation
* Web Deployment
* Prediction Logging

---

## ✨ Features

* 🤖 AI-powered loan eligibility prediction
* ⚡ Real-time prediction using an optimized XGBoost model
* 📊 Interactive analytics dashboard
* 📈 Exploratory Data Analysis (EDA) visualizations
* 📝 Multi-step loan application form
* 📋 Prediction history and audit logs
* 🎯 Confidence probability for every prediction
* 📱 Fully responsive Glassmorphism-inspired UI
* 🗄 SQLite database integration with SQLAlchemy ORM
* 🚀 Load-tested using Locust
* 🔒 Clean Flask backend architecture

---

## 🛠 Technology Stack

### Frontend

* HTML5
* CSS3
* JavaScript
* Responsive Design
* Glassmorphism UI

### Backend

* Python
* Flask

### Database

* SQLite
* SQLAlchemy ORM

### Machine Learning

* XGBoost
* Scikit-learn
* Pandas
* NumPy
* Imbalanced-learn (SMOTE)

### Data Visualization

* Matplotlib
* Seaborn

### Performance Testing

* Locust

---

## 📂 Project Structure

```text
Smart-Lender/
│
├── Dataset/
│   └── loan_prediction.csv
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── images/
│       └── plots/
│
├── templates/
│   ├── layout.html
│   ├── home.html
│   ├── predict.html
│   ├── submit.html
│   └── history.html
│
├── app.py
├── models.py
├── train_models.py
├── tune_xgb.py
├── locustfile.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/kvnvk9955-stack/Smart-Lender.git
```

### Navigate into the project directory

```bash
cd Smart-Lender
```

### Create a virtual environment (Optional)

```bash
python -m venv venv
```

### Activate the virtual environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Flask development server:

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 🤖 Machine Learning Pipeline

The machine learning workflow consists of the following stages:

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Missing Value Handling
5. Feature Engineering
6. Data Scaling
7. Class Balancing using SMOTE
8. Model Training
9. Hyperparameter Tuning
10. Model Evaluation
11. Model Serialization
12. Flask Deployment

---

## 📊 Model Performance

| Model              |   Accuracy |
| ------------------ | ---------: |
| XGBoost Classifier | **80.29%** |

### Model Features

* Hyperparameter tuning using Random Search
* SMOTE balancing for imbalanced datasets
* StandardScaler preprocessing
* Probability estimation for prediction confidence

---

## 📝 Input Parameters

The prediction model evaluates applicants based on:

* Gender
* Marital Status
* Number of Dependents
* Education
* Self Employment Status
* Applicant Income
* Co-applicant Income
* Loan Amount
* Loan Amount Term
* Credit History
* Property Area

---

## 🎯 Prediction Output

The application provides:

* ✅ Loan Approval Prediction
* 📈 Confidence Probability Score
* 📝 Prediction Record Storage
* 📊 Historical Prediction Logs

---

## 📊 Exploratory Data Analysis

The project includes several visualization techniques, including:

* Univariate Analysis
* Bivariate Analysis
* Multivariate Analysis
* Correlation Heatmaps
* Feature Distribution Charts
* Model Performance Comparisons

---

## 🧪 Performance Testing

Application performance was evaluated using **Locust**, including:

* Concurrent User Simulation
* Request Throughput Analysis
* Response Time Monitoring
* Load Stability Testing

---

## 🚀 Future Enhancements

* User Authentication & Authorization
* Admin Dashboard
* Cloud Database Integration
* Docker Containerization
* REST API Support
* Explainable AI (SHAP/LIME)
* PDF Report Generation
* Email Notifications
* Loan EMI Calculator
* Multi-language Support

---

## 📚 Libraries Used

* Flask
* XGBoost
* Scikit-learn
* Pandas
* NumPy
* SQLAlchemy
* Imbalanced-learn
* Matplotlib
* Seaborn
* Joblib
* SQLite

---

## 🎓 Learning Outcomes

This project demonstrates practical experience in:

* Machine Learning Model Development
* Data Preprocessing
* Hyperparameter Optimization
* Exploratory Data Analysis
* Flask Web Development
* SQLAlchemy ORM
* Database Design
* Responsive Frontend Development
* Model Deployment
* Performance Testing
* Version Control with Git & GitHub

---

## 👨‍💻 Developd By

**Venkata Naga Vamsi Krishna Kolli**
**Akshaya Vadduri**

* GitHub: https://github.com/kvnvk9955-stack
* Demo: https://drive.google.com/file/d/1O5voAYSTYKgEXtLlMTMfywgIh7nz5JMN/view?usp=sharing
* Project Website link: https://kvnvk.pythonanywhere.com

---
