from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import datetime
import time
import logging
import random

app = Flask(__name__)

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Application started: Monitoring and Logging in MLOps")

# Load and train model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "logs/monitored_model.pkl")
logging.info("Model trained and saved successfully.")

# Simulate initial log entries
for i in range(3):
    latency = round(random.uniform(10, 20), 2)
    logging.info(
        f"Simulated log entry {i+1}: "
        f"Input={[random.random() for _ in range(4)]} | "
        f"Output={random.choice([0, 1, 2])} | "
        f"Latency={latency}ms"
    )

@app.route("/", methods=["GET"])
def home():
    """
    Minimal home route to confirm service availability.
    """
    logging.info("Home route accessed.")
    return "MLOps Monitoring App is running."

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle model predictions and log request details.
    Includes input validation to detect malformed requests.
    """
    start_time = time.time()

    # 1) Ensure the request is JSON
    if not request.is_json:
        logging.error("Invalid input: request is not JSON")
        return jsonify({"error": "Invalid input data. Request must be JSON."}), 400

    data = request.get_json(silent=True)

    # 2) Handle malformed or empty JSON
    if data is None:
        logging.error("Invalid input: malformed JSON or empty request body")
        return jsonify({"error": "Invalid input data. Malformed JSON."}), 400

    # 3) Validate 'features'
    features = data.get("features")

    if not isinstance(features, list) or len(features) != 4:
        logging.error(f"Invalid input: expected 'features' list of length 4. Received: {data}")
        return jsonify({"error": "Invalid input data. Expecting 'features' with 4 numeric values."}), 400

    # 4) Validate numeric values
    if not all(isinstance(x, (int, float)) for x in features):
        logging.error(f"Invalid input: non-numeric values in features. Received: {features}")
        return jsonify({"error": "Invalid input data. Features must contain only numeric values."}), 400

    # 5) Normal prediction flow
    prediction = int(model.predict([features])[0])
    latency = round((time.time() - start_time) * 1000, 2)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Simulated correctness (as in your previous assignment)
    correct = bool(prediction == iris.target[0])

    logging.info(
        f"Prediction made | Input={features} | Output={prediction} | "
        f"Correct={correct} | Latency={latency}ms"
    )

    return jsonify({
        "prediction": prediction,
        "timestamp": timestamp,
        "latency_ms": latency
    })

@app.route("/monitor", methods=["GET"])
def monitor():
    """
    Parse log file and show summary metrics.
    """
    try:
        with open("logs/app.log", "r") as f:
            lines = f.readlines()
            recent_logs = lines[-10:]
            total_predictions = sum(1 for line in lines if "Prediction made" in line)
            total_invalid_requests = sum(1 for line in lines if "Invalid input" in line)
    except FileNotFoundError:
        recent_logs = []
        total_predictions = 0
        total_invalid_requests = 0

    logging.info("Monitor endpoint accessed.")

    return jsonify({
        "total_predictions": total_predictions,
        "total_invalid_requests": total_invalid_requests,
        "recent_activity": recent_logs
    })

@app.route("/health", methods=["GET"])
def health():
    """
    Return a simple status report based on recent activity.
    """
    try:
        with open("logs/app.log", "r") as f:
            lines = f.readlines()
            # If there are no logs yet, consider healthy.
            if not lines:
                healthy = True
            else:
                # If the last log line includes an ERROR, mark as degraded (simple heuristic).
                healthy = "ERROR" not in lines[-1]
    except Exception:
        healthy = False

    status = "healthy" if healthy else "degraded"
    logging.info(f"Health endpoint accessed: Status={status}")

    return jsonify({
        "status": status,
        "checked_at": datetime.datetime.now().isoformat()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
