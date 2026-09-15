# ANN Churn Classification

A Deep Learning project that predicts whether a bank customer is likely to leave (churn) based on customer and account information. The project uses an Artificial Neural Network (ANN) with data preprocessing, encoding, feature scaling, model training, and a prediction interface.

## Tech Stack

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Project Workflow

1. Data Collection
2. Data Preprocessing
3. Categorical Feature Encoding
4. Feature Scaling
5. ANN Model Building
6. Model Training
7. Model Evaluation
8. Customer Churn Prediction

## Features

- Predicts customer churn using an ANN
- Handles categorical features using encoding
- Uses feature scaling for better model performance
- Saves the trained model for future predictions
- Provides a prediction application using Streamlit

## Project Structure

```text
ANN-Churn-Classification/
│
├── Churn_Modelling.csv
├── app.py
├── model.h5
├── label_encoder_gender.pkl
├── onehot_encoder_geo.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
