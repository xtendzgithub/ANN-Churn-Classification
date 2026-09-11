 ## INTEGRATING ANN MODEL WITH STREAMLIT APPP   .............


import numpy as np 
import streamlit as st 
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler , LabelEncoder ,OneHotEncoder
import pandas as pd 
import pickle



# Load the trained model....


# ==============================
# Load trained model
# ==============================

model = tf.keras.models.load_model('model.h5')


# ==============================
# Load encoders and scaler
# ==============================

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# ==============================
# Streamlit app
# ==============================

st.title("Customer Churn Prediction")


# ==============================
# User input
# ==============================

geography = st.selectbox(
    'Geography',
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    'Gender',
    label_encoder_gender.classes_
)

age = st.slider(
    'Age',
    18,
    92
)

balance = st.number_input(
    'Balance',
    min_value=0.0
)

credit_score = st.number_input(
    'Credit Score',
    min_value=0
)

estimated_salary = st.number_input(
    'Estimated Salary',
    min_value=0.0
)

tenure = st.slider(
    'Tenure',
    0,
    10
)

num_of_products = st.slider(
    'Number of Products',
    1,
    4
)

has_cr_card = st.selectbox(
    'Has Credit Card',
    [0, 1]
)

is_active_member = st.selectbox(
    'Is Active Member',
    [0, 1]
)


# ==============================
# Prepare input data
# ==============================

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [
        label_encoder_gender.transform([gender])[0]
    ],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


# ==============================
# One-hot encode Geography
# ==============================

geo_encoded = onehot_encoder_geo.transform(
    input_data[['Geography']]
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(
        ['Geography']
    )
)


# ==============================
# Combine input and encoded data
# ==============================

# Original Geography column remove karo
input_data = input_data.drop(
    columns=['Geography']
)

# Geography encoded columns sirf ek baar add karo
input_data = pd.concat(
    [
        input_data.reset_index(drop=True),
        geo_encoded_df.reset_index(drop=True)
    ],
    axis=1
)


# ==============================
# Match training column order
# ==============================

input_data = input_data[
    scaler.feature_names_in_
]


# ==============================
# Scale input data
# ==============================

input_data_scaled = scaler.transform(
    input_data
)


# ==============================
# Predict churn
# ==============================

if st.button("Predict Churn"):

    prediction = model.predict(
        input_data_scaled,
        verbose=0
    )

    prediction_proba = prediction[0][0]

    if prediction_proba > 0.5:
        st.error(
            f"Customer may churn. "
            f"Probability: {prediction_proba:.2%}"
        )
    else:
        st.success(
            f"Customer may not churn. "
            f"Churn Probability: {prediction_proba:.2%}"
        )