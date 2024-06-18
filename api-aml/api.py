# import pickle
# import os
# import numpy as np
# import pandas as pd
# import streamlit as st

# def load_model(file_path):
#     with open(file_path, 'rb') as f:
#         model = pickle.load(f)
#     return model

# def preprocess_data(data):
#     data = data.drop(['Unnamed: 0','step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
#     data = data.fillna(data.median())   
#     data = pd.get_dummies(data, columns=['type'])
#     numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
#     return data

# def scale_numerical_cols(data):
#     mean = np.mean(data, axis=0)
#     std = np.std(data, axis=0)
#     return (data - mean) / std

# def app():
#     st.title('Filtering Fraudulent Transactions')
#     st.write('Filtering out fraudulent transactions using a machine learning model')
    
#     # Load the model
#     model_path = 'model.pkl'
#     model = load_model(model_path)
   
#     # File uploader for the input 
#     uploaded_file = st.file_uploader('Upload your data:', type=['csv', 'xlsx'])
#     if uploaded_file is not None:
#         data = pd.read_csv(uploaded_file)
        
#         # Preprocess the data
#         data = preprocess_data(data)
        
#         # Make predictions for data
#         predictions = model.predict(data)
#         predictions = [1 if p == 'fraud' else 0 for p in predictions]
        
#         result_df = pd.DataFrame({'Row': range(len(predictions)), 'Prediction': predictions})
#         result_df['Prediction'] = result_df['Prediction'].apply(lambda x: 'Fraudulent' if x == 1 else 'Not fraudulent')
#         st.table(result_df)

# if __name__=='__main__':
#     app()

import pickle
import numpy as np
import pandas as pd
import streamlit as st

def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    # print(model)
    return model

def preprocess_data(data):
    # Drop unnecessary columns
    data = data.drop(['Unnamed: 0', 'step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
    # print(data)

    # Fill missing values with median
    data = data.fillna(data.select_dtypes(include=['number']).median())
    
    # One-hot encode categorical columns
    data = pd.get_dummies(data, columns=['type'])
    
    return data

def scale_numerical_cols(data):
    # Scale numerical columns
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

def app():
    st.title('Money Laundering Detection')
    st.write('Identifying the Fraudural transactions using a machine learning model')
    
    # Load the model
    model_path = 'model.pkl'
    model = load_model(model_path)
    
    # File uploader for the input 
    uploaded_file = st.file_uploader('Upload your data:', type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Preprocess the data
        data = preprocess_data(data)
        
        # Scale numerical columns
        data = scale_numerical_cols(data)
        
        # print(model)
        # Make predictions for data
        # print(model.columns)
        predictions = model['isFraud']
        predictions = ['Fraud' if p == 1 else 'Not fraud' for p in predictions]
        
        result_df = pd.DataFrame({'Row': range(len(predictions)), 'Prediction': predictions})
        st.table(result_df)

if __name__=='__main__':
    app()
