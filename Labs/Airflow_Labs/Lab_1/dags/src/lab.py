import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import base64
import os

def load_data():
    """
    Loads the diabetes dataset from CSV, serializes it,
    and returns a base64-encoded string (XCom-safe).
    """
    csv_path = os.path.join(os.path.dirname(__file__), "../data/diabetes.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with shape: {df.shape}")
    serialized = pickle.dumps(df)
    return base64.b64encode(serialized).decode("ascii")


def data_preprocessing(data_b64: str):
    """
    Deserializes data, scales features, splits into train/test,
    and returns serialized splits.
    """
    df = pickle.loads(base64.b64decode(data_b64))

    # Target column
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    payload = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.values,
        "y_test": y_test.values,
    }

    serialized = pickle.dumps(payload)
    return base64.b64encode(serialized).decode("ascii")


def build_save_model(data_b64: str, model_path: str):
    """
    Trains a Linear Regression model and saves it to disk.
    Returns the model path.
    """
    payload = pickle.loads(base64.b64decode(data_b64))
    X_train = payload["X_train"]
    y_train = payload["y_train"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model
    save_path = os.path.join(os.path.dirname(__file__), f"../../{model_path}")
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to: {save_path}")
    return model_path


def load_model_evaluate(model_path: str, data_b64: str):
    """
    Loads the saved model and evaluates it on the test set.
    Prints MSE and R2 score.
    """
    payload = pickle.loads(base64.b64decode(data_b64))
    X_test = payload["X_test"]
    y_test = payload["y_test"]

    load_path = os.path.join(os.path.dirname(__file__), f"../../{model_path}")
    with open(load_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE:  {mse:.4f}")
    print(f"R2:   {r2:.4f}")
    print(f"Coefficients: {model.coef_}")

    return {"mse": mse, "r2": r2}

    Links:
    http://localhost:8080/dags/Airflow_Lab1_Diabetes/runs
    http://localhost:8080/dags/Airflow_Lab1_Diabetes/events
    http://localhost:8080/dags/Airflow_Lab1_Diabetes

Airflow Lab 1 — Diabetes Prediction Pipeline
For this lab, I swapped out the original dataset and model to make it my own — I used the Diabetes dataset and built a Linear Regression pipeline instead of KMeans clustering. Honestly, getting Airflow running in Docker took more troubleshooting than expected (volume mounts, missing packages, wrong container names) but it was a solid learning experience overall.
The pipeline has 4 tasks: loading the data, preprocessing it, training the model, and evaluating it — all running sequentially in Airflow.
Results
DAG Run — All Tasks Successful
Graph View
Model Evaluation Logs
How to Run
bashdocker compose up -d
docker compose exec airflow-apiserver airflow dags trigger Airflow_Lab1_Diabetes
Then open http://localhost:8080 to watch it run.
Tech Stack

Apache Airflow 3.1.7 · Docker · Python 3.12 · scikit-learn · pandas
