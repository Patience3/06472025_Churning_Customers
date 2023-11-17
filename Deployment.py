import streamlit as st
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib
import pickle 

class InvalidChoiceError(Exception):
    pass
# Initialize session state
if 'user_prediction' not in st.session_state:
    st.session_state.user_prediction = None

if 'user_confidence' not in st.session_state:
    st.session_state.user_confidence = None

# Load your trained model
model = keras.models.load_model('best_model.h5')



# Function to preprocess user input
def preprocess_user_input(user_input):
    # Preprocess user input as needed (make sure it matches the preprocessing of your training data)
    # Example: Scaling the input using StandardScaler
    #Load the scaler using pickle
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
        user_input_scaled = scaler.transform(user_input)
    return user_input_scaled

# Function to predict with confidence
def predict_with_confidence(model, X):
    predictions = model.predict(X)[0][0]
    confidences = predictions
    return predictions, confidences

# Map 'Yes' to 1 and 'No' to 0
def map_yes_no_to_binary(value, feature_name=None):
    try:
        if value is None:
            raise InvalidChoiceError(f"Please recheck your response for {feature_name or 'the feature'}. It cannot be None.")
        else:
            return 1 if value.lower() == 'yes' else 0
    except InvalidChoiceError as e:
        st.error(str(e))
        st.stop()

# Example usage
#online_security_no_binary = map_yes_no_to_binary(online_security_no)


# Streamlit app
st.title('Churn Prediction')

# User input form
tenure = st.number_input('Please enter the value of your Tenure:', value=0, step=1)
monthly_charges = st.number_input('What is your Monthly Charges:', value=0.0, step=0.1)
online_security_no = st.radio('Do you have Online Security:', [None, 'Yes', 'No'])
internet_service_no = st.radio('Do you have Internet Service:', [None, 'Yes', 'No'])
payment_method_electronic_check = st.radio('Do you use the Electronic check for payment?:', [None, 'Yes', 'No'])
internet_service_fiber_optic = st.radio('Is your internet suplies fibre optic?:', [None, 'Yes', 'No'])
paperless_billing_yes = st.radio('Do you make use of paperless billing?', [None, 'Yes', 'No'])
tech_support_no = st.radio('Do you have Tech support:', [None, 'Yes', 'No'])
dependents_yes = st.radio('Do you have dependents:', [None, 'Yes', 'No'])
contract_month_to_month = st.radio('Is your Contract Month-to-Month:', [None, 'Yes', 'No'])
device_protection_no = st.radio('Do you enjoy Device Protection :', [None, 'Yes', 'No'])
online_backup_no = st.radio('Do you have Online Backup:', [None, 'Yes', 'No'])
senior_citizen_yes = st.radio('Are you a senior citizen?:', [None, 'Yes', 'No'])
tech_support_yes = st.radio('Do you enjy Tech Support:', [None, 'Yes', 'No'])
internet_service_dsl = st.radio('Is your internet service DSL?:', [None, 'Yes', 'No'])
partner_yes = st.radio('Do you have a partner:', [None, 'Yes', 'No'])
online_security_yes = st.radio('Online Security (Yes):', [None, 'Yes', 'No'])
phone_service_no = st.radio('Do you have a phone service?:', [None, 'Yes', 'No'])
contract_one_year = st.radio('Do you use a one year contract?:', [None, 'Yes', 'No'])

if st.button('Predict'):
    # Map 'Yes' and 'No' to 1 and 0
    online_security_no_binary = map_yes_no_to_binary(online_security_no)
    internet_service_no_binary = map_yes_no_to_binary(internet_service_no)
    payment_method_electronic_check_binary = map_yes_no_to_binary(payment_method_electronic_check)
    internet_service_fiber_optic_binary = map_yes_no_to_binary(internet_service_fiber_optic)
    paperless_billing_yes_binary = map_yes_no_to_binary(paperless_billing_yes)
    tech_support_no_binary = map_yes_no_to_binary(tech_support_no)
    dependents_yes_binary = map_yes_no_to_binary(dependents_yes)
    contract_month_to_month_binary = map_yes_no_to_binary(contract_month_to_month)
    device_protection_no_binary = map_yes_no_to_binary(device_protection_no)
    online_backup_no_binary = map_yes_no_to_binary(online_backup_no)
    senior_citizen_yes_binary = map_yes_no_to_binary(senior_citizen_yes)
    tech_support_yes_binary = map_yes_no_to_binary(tech_support_yes)
    internet_service_dsl_binary = map_yes_no_to_binary(internet_service_dsl)
    partner_yes_binary = map_yes_no_to_binary(partner_yes)
    online_security_yes_binary = map_yes_no_to_binary(online_security_yes)
    phone_service_no_binary = map_yes_no_to_binary(phone_service_no)
    contract_one_year_binary = map_yes_no_to_binary(contract_one_year)

    # Preprocess user input
    user_input_array = preprocess_user_input([[
        tenure, monthly_charges, online_security_no_binary, internet_service_no_binary,
        payment_method_electronic_check_binary, internet_service_fiber_optic_binary,
        paperless_billing_yes_binary, tech_support_no_binary, dependents_yes_binary,
        contract_month_to_month_binary, device_protection_no_binary, online_backup_no_binary,
        senior_citizen_yes_binary, tech_support_yes_binary, internet_service_dsl_binary,
        partner_yes_binary, online_security_yes_binary, phone_service_no_binary, contract_one_year_binary
    ]])

    # Predict and get confidences for the user input
    user_prediction, user_confidence = predict_with_confidence(model, user_input_array)

    if user_prediction >= 0.5:
        confidence_message = f'Confidence: {user_confidence:.2f}'
    else:
       confidence_message = f'Confidence: {1 - user_confidence:.2f}'

    # Display the prediction and adjusted confidence
    prediction_label = "Likely to Churn" if user_prediction >= 0.5 else "Not Likely to Churn"
    st.success(f'Prediction: {prediction_label}')
    st.info(confidence_message)
def get_user_prediction():
    return {'user_prediction': None, 'user_confidence': None}

user_state = get_user_prediction()
# Reset button
if st.button('Reset Records'):
    user_state['user_prediction'] = None
    user_state['user_confidence'] = None

# Display previous prediction if available
if user_state['user_prediction'] is not None:
    st.warning('Previous Prediction:')
    previous_prediction_label = "Likely to Churn" if user_state['user_prediction'] >= 0.5 else "Not Likely to Churn"
    previous_confidence_message = f'Confidence: {user_state["user_confidence"]:.2f}'
    st.write(f'Prediction: {previous_prediction_label}')
    st.write(previous_confidence_message)