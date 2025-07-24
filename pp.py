import subprocess
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn

# -------- Start MLflow Server Automatically --------
def start_mlflow():
    try:
        # Check if MLflow is already running on port 5000
        subprocess.check_output("netstat -ano | findstr :5000", shell=True)
        print("MLflow server is already running on port 5000.")
    except subprocess.CalledProcessError:
        print("Starting MLflow server...")
        subprocess.Popen(
            ["mlflow", "server", "--host", "127.0.0.1", "--port", "5000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            shell=True
        )
        time.sleep(5)  # Give MLflow some time to start

start_mlflow()
# ---------------------------------------------------

# Load dataset
df = pd.read_csv("Rainfall.csv")
df.columns = df.columns.str.strip()
df = df.drop(columns=['day'])

df['winddirection'] = df['winddirection'].fillna(df['winddirection'].mode()[0])
df['windspeed'] = df['windspeed'].fillna(df['windspeed'].median())
df['rainfall'] = df['rainfall'].map({"yes": 1, "no": 0})

df = df.drop(columns=['maxtemp', 'temparature', 'mintemp'])

# Balance data
df_majority = df[df['rainfall'] == 1]
df_minority = df[df['rainfall'] == 0]
df_majority_downsampled = df_majority.sample(len(df_minority), random_state=42)
df_downsampled = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42)

# Train/Test Split
X = df_downsampled.drop(columns=['rainfall'])
y = df_downsampled['rainfall']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
param_grid_rf = {
    "n_estimators": [100],
    "max_features": ['sqrt'],
    "max_depth": [10],
    "min_samples_split": [2],
    "min_samples_leaf": [1]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=1)
grid_search_rf.fit(X_train, y_train)

best_model = grid_search_rf.best_estimator_
y_pred = best_model.predict(X_test)
report_dict = classification_report(y_test, y_pred, output_dict=True)

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Rainfall1")

with mlflow.start_run() as run:
    mlflow.log_params(grid_search_rf.best_params_)
    mlflow.log_metrics({
        "accuracy": report_dict['accuracy'],
        "recall_0": report_dict['0']['recall'],
        "recall_1": report_dict['1']['recall'],
        "f1_macro": report_dict['macro avg']['f1-score']
    })
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="rainfall-prediction-production"
    )

# Load model from registry
prod_model_uri = "models:/rainfall-prediction-production/1"
loaded_model = mlflow.sklearn.load_model(prod_model_uri)

# Prediction Input
input_data = pd.DataFrame([{
    "pressure": 1015.9,
    "dewpoint": 19.9,
    "humidity": 95,
    "cloud": 81,
    "sunshine": 0.0,
    "winddirection": 40.0,
    "windspeed": 13.7
}])

prediction = loaded_model.predict(input_data)
print("Prediction Result:", "Rainfall" if prediction[0] == 1 else "No Rainfall")
