from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and feature names
model = joblib.load("model/fake_profile_detector.pkl")
features = joblib.load("model/model_features.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([data], columns=features)
        input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_label = "Fake Profile" if prediction == 1 else "Real Profile"
        
        # Return JSON response
        return jsonify({"prediction": prediction_label})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
