import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load and preprocess
df = pd.read_csv("Rainfall.csv")
df.columns = df.columns.str.strip()
df['rainfall'] = df['rainfall'].map({'yes': 1, 'no': 0})
df['winddirection'] = df['winddirection'].fillna(df['winddirection'].mode()[0])
df['windspeed'] = df['windspeed'].fillna(df['windspeed'].median())
df = df.drop(columns=['day', 'maxtemp', 'temparature', 'mintemp'])  # Drop extra cols

# 2. Features and target
X = df.drop("rainfall", axis=1)
y = df["rainfall"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. Log & register with MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Rainfall1")

with mlflow.start_run():
    mlflow.log_metric("accuracy", accuracy_score(y_test, model.predict(X_test)))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="rainfall-prediction-production"
    )
