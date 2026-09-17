
# Credit Card Fraud Detection

## Project Overview

This project uses Machine Learning to classify credit card transactions as normal or fraudulent.

## Objective

To build a classification model that predicts whether a credit card transaction is normal or fraudulent.

## Dataset

Credit Card Transactions Dataset

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

## Machine Learning Models

- Logistic Regression
- Support Vector Classifier
- Decision Tree Classifier
- K-Nearest Neighbors Classifier
- Random Forest Classifier

## Project Workflow

1. Data Collection
2. Data Analysis
3. Data Preprocessing
4. Feature Scaling
5. Classification
6. Train and Test
7. Model Evaluation
8. Streamlit Application
9. GitHub Deployment

## Data Preprocessing

- Missing value checking
- Duplicate checking
- Feature and target separation
- Train-test split
- StandardScaler

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Streamlit Application

The application accepts transaction details and predicts whether the transaction is normal or fraudulent.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Project Files

- creditcard.csv – Dataset
- fraud_detection.ipynb – Data analysis and model training
- app.py – Streamlit application
- model.pkl – Trained model
- scaler.pkl – Feature scaler
- requirements.txt – Required libraries
- README.md – Project documentation

## Conclusion

This project demonstrates the use of Machine Learning for credit card fraud detection through data analysis, classification, model evaluation, and Streamlit deployment.