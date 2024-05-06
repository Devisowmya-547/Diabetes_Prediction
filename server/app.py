from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
with open('gradient_boosting_classifier.pkl', 'rb') as file:
    gb_classifier = pickle.load(file)

# Preprocess input data
def preprocess_input(input_data):
    # Convert input into DataFrame
    input_df = pd.DataFrame([input_data])

    # Manually encode categorical variables
    input_df['gender'] = input_df['gender'].map({'Male': 0, 'Female': 1})
    input_df['smoking_history'] = input_df['smoking_history'].map({'never': 0, 'current': 1, 'former': 2})

    # Scale numerical features
    scaler = StandardScaler()
    scaled_input = scaler.fit_transform(input_df)

    return scaled_input

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    input_data = request.form.to_dict()
    
    # Preprocess input data
    processed_input = preprocess_input(input_data)

    # Make prediction
    prediction = gb_classifier.predict(processed_input)

    # Display result
    if prediction[0] == 1:
        result = "Positive for Diabetes"
    else:
        result = "Negative for Diabetes"

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
