from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__, static_folder='templates/static')
CORS(app)

# Load model and feature columns
model = joblib.load("model.pkl")
feature = pd.read_csv('feature_columns.csv')
feature_columns = feature.values.flatten().tolist()

# Load unique targets
with open("unique_targets.txt", "r") as f:
    unique_targets = [line.strip() for line in f.readlines()]

def preprocess_and_predict(user_input, model, feature_columns):
    # Convert user input to a DataFrame
    user_df = pd.DataFrame([user_input], columns=feature_columns)
    
    # Ensure all required columns exist
    for col in feature_columns:
        if col not in user_df.columns:
            user_df[col] = 0  # Set missing columns to 0
    
    # Ensure the column order matches the model's expected input
    user_df = user_df[feature_columns]
    
    # Make prediction
    prediction = model.predict(user_df)[0]  # Predict the target class
    probabilities = model.predict_proba(user_df)[0]  # Predict probabilities
    return prediction, probabilities

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Extract user input from the form, using default values for None
        user_input = {
            "Age": int(request.form.get("Age", 0)),
            "Birth Weight": float(request.form.get("Birth Weight", 0.0)),
            "Blood Glucose Levels": float(request.form.get("Blood Glucose Levels", 0.0)),
            "Blood Pressure": float(request.form.get("Blood Pressure", 0.0)),
            "BMI": float(request.form.get("BMI", 0.0)),
            "Cholesterol Levels": float(request.form.get("Cholesterol Levels", 0.0)),
            "Digestive Enzyme Levels": float(request.form.get("Digestive Enzyme Levels", 0.0)),
            "Insulin Levels": float(request.form.get("Insulin Levels", 0.0)),
            "Pulmonary Function": float(request.form.get("Pulmonary Function", 0.0)),
            "Weight Gain During Pregnancy": float(request.form.get("Weight Gain During Pregnancy", 0.0)),
        }

        # Predict using the model
        predicted_index, probabilities = preprocess_and_predict(user_input, model, feature_columns)

        # Map the predicted index to the target name
        predicted_diabetes = unique_targets[predicted_index]

        # Get the probability of the predicted class
        probability_percentage = probabilities[predicted_index] * 100

        # Render the result page with the prediction and probability
        return render_template('result.html', 
                               probability=f"{probability_percentage:.2f}%", 
                               predicted_diabetes=predicted_diabetes)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
