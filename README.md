Ni# Deep Neural Network for Fraudulent Transaction Detection

## Project Overview

This project develops an AI-powered fraud detection system using a Deep Neural Network (DNN) to classify credit card transactions as genuine or potentially fraudulent.

The model uses ReLU activation functions in the hidden layers and Sigmoid activation in the output layer.

## Objectives

- Detect potentially fraudulent transactions
- Handle highly imbalanced transaction data
- Build a Deep Neural Network using ReLU and Sigmoid
- Evaluate the model using multiple performance metrics
- Provide an interactive Streamlit dashboard
- Allow transaction prediction through sample data and CSV upload

## Model Architecture

Input Layer  
↓  
Dense Layer - 64 neurons - ReLU  
↓  
Dense Layer - 32 neurons - ReLU  
↓  
Output Layer - 1 neuron - Sigmoid

## Technologies Used

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Plotly
- Streamlit
- Joblib

## Dataset

Credit Card Fraud Detection dataset.

The dataset contains approximately 284,807 transactions, including 492 fraudulent transactions.

The dataset is highly imbalanced, so class weights were used during model training.

## Model Performance

| Metric | Result |
|---|---:|
| Accuracy | 99.56% |
| Precision | 25.08% |
| Recall | 83.16% |
| F1 Score | 38.54% |
| ROC-AUC | 95.72% |
| PR-AUC | 70.56% |

## Application Features

- Interactive dashboard
- Transaction statistics
- Fraud prediction
- Fraud probability
- Risk classification
- CSV transaction upload
- Prediction result download
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Training performance graphs

## How to Run

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL