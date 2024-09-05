from flask import Flask, request, render_template
import numpy as np
import joblib
import sklearn

# Ensure scikit-learn version is correct
print(f"scikit-learn version: {sklearn.__version__}")

# Load models
try:
    dtr = joblib.load('dtr.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# Flask app
app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract and convert input features
            Year = float(request.form['Year'])
            average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
            pesticides_tonnes = float(request.form['pesticides_tonnes'])
            avg_temp = float(request.form['avg_temp'])
            Area = request.form['Area']  # Area should be a string for categorical data
            Item = request.form['Item']  # Item should be a string for categorical data
            destination = request.form['destination']  # Destination field

            # Prepare features for prediction
            features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]],
                                dtype=object)

            # Transform features
            transformed_features = preprocessor.transform(features)

            # Make prediction
            prediction = dtr.predict(transformed_features)

            # Calculate storage capacity
            storage_capacity = prediction[0] / 30

            # Format the prediction and storage capacity
            prediction_text = f"{prediction[0]:.2f}"
            storage_capacity_text = f"{storage_capacity:.2f}"

            return render_template('index.html',
                                   prediction=prediction_text,
                                   storage_capacity=storage_capacity_text,
                                   destination=destination)  # Pass destination to HTML
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('index.html',
                                   prediction="An error occurred during prediction.")

if __name__ == "__main__":
    app.run(debug=True)
