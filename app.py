from flask import Flask, request, render_template
import pandas as pd
import mlflow
import mlflow.sklearn

# Initialize Flask app
app = Flask(__name__)

# Set MLflow tracking
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")  # Set this before anything
mlflow.set_experiment("Rainfall1")

# Load the production model
production_model_name = "Random Forest Model Data"
prod_model_uri = "models:/Random Forest Model Data/1"
# If using version
loaded_model = mlflow.sklearn.load_model("models:/rainfall-prediction-production/1")


# Feature names
feature_names = ['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

@app.route('/')
def home():
    return render_template('index.html', prediction_result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        input_data = [
            float(request.form['pressure']),
            float(request.form['dewpoint']),
            float(request.form['humidity']),
            float(request.form['cloud']),
            float(request.form['sunshine']),
            float(request.form['winddirection']),
            float(request.form['windspeed']),
        ]

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Make prediction
        prediction = loaded_model.predict(input_df)

        # Convert to label
        result = "Rainfall" if prediction[0] == 1 else "No Rainfall"

        return render_template('index.html', prediction_result=result)
    except Exception as e:
        return render_template('index.html', prediction_result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=8080)
