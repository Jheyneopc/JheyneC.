import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

MODEL_PATH = "model/best_model.pkl"
LOG_PATH = "model/performance_log.csv"

# Timestamp for traceability
ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Load dataset and simulate "new incoming data" via random split
X, y = load_iris(return_X_y=True)
seed = int(np.random.randint(0, 10_000))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

# Load current best model (if missing, create a baseline one)
if os.path.exists(MODEL_PATH):
    old_model = joblib.load(MODEL_PATH)
else:
    old_model = RandomForestClassifier(random_state=42)
    old_model.fit(X_train, y_train)
    joblib.dump(old_model, MODEL_PATH)

old_acc = accuracy_score(y_test, old_model.predict(X_test))

# Train a new candidate model (different params / seed)
new_model = RandomForestClassifier(n_estimators=150, random_state=seed)
new_model.fit(X_train, y_train)
new_acc = accuracy_score(y_test, new_model.predict(X_test))

print(f"Old Accuracy: {old_acc:.3f} | New Accuracy: {new_acc:.3f}")

# Replace model only if improved
improved = new_acc > old_acc
if improved:
    joblib.dump(new_model, MODEL_PATH)
    print("Model improved and updated.")
else:
    print("New model not better. Keeping the previous one.")

# Load or create log (with timestamp)
if os.path.exists(LOG_PATH):
    log_df = pd.read_csv(LOG_PATH)
else:
    log_df = pd.DataFrame(columns=["timestamp", "event", "old_accuracy", "new_accuracy", "improved", "seed"])

new_row = {
    "timestamp": ts,
    "event": "KaizenUpdate",
    "old_accuracy": round(float(old_acc), 6),
    "new_accuracy": round(float(new_acc), 6),
    "improved": bool(improved),
    "seed": seed,
}

log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
log_df.to_csv(LOG_PATH, index=False)

print(f"Log updated at {ts}")
print(f"Model log saved to {LOG_PATH}")
