# 📊 Telco Customer Churn Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]
(https://churn-prediction-mrafc9cck3aagyqzgcqy3s.streamlit.app/)

A machine learning web app that predicts whether a telecom customer is likely to churn,
and recommends a retention action.

## 🎯 Problem
Telecom companies lose ~$300 per churned customer. This model identifies customers 
likely to cancel so the retention team can intervene early.

## 📊 Dataset
- **Source**: IBM Telco Customer Churn (7,043 customers, 20 features)
- **Target**: Binary — Churn (Yes/No)
- **Class balance**: 73.5% No Churn, 26.5% Churn

## 🏆 Results
| Model                      | F1 (Churn) | Recall | ROC-AUC |
|----------------------------|------------|--------|---------|
| Gradient Boosting (tuned)  | 0.63       | 0.72   | 0.843   |
| Logistic Regression        | 0.61       | 0.61   | 0.841   |
| Random Forest              | 0.59       | 0.59   | 0.821   |
| Baseline (always No Churn) | 0.00       | 0.00   | ——  |

> Threshold tuned to 0.343 (from default 0.5) to maximise Recall — catches 72% of churners.

## 🔍 Key Findings
- Month-to-month contracts are the #1 churn driver (38% feature importance)
- Low tenure customers churn most — new customers need better onboarding
- Fiber optic customers without online security are the highest risk segment
- Tuning the decision threshold from 0.5 → 0.343 rescued 75 more churners per 1,409 customers

## 🛠 Tech Stack
Python · scikit-learn · Pandas · Streamlit · Matplotlib · joblib

## 🚀 Run Locally
```bash
git clone https://github.com/Chandan-cmy/churn-prediction
cd churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure

churn-prediction/
├── data/ # raw + processed data
├── notebooks/ # EDA + modelling notebooks
│ ├── 01_EDA.ipynb
│ └── 02_Modelling.ipynb
├── plots/ #saved graphs
├── models/ # saved model
│ └── best_model.pkl
├── app.py # Streamlit app
└── requirements.txt