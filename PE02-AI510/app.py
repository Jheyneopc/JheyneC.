from flask import Flask, request, jsonify
import joblib
import numpy as np

# PE02 imports (runtime info)
import platform
import socket
from importlib.metadata import version as pkg_version

app = Flask(__name__)

# Load trained model
model = joblib.load("model/iris_model.pkl")

# Class mapping
species = {0: "setosa", 1: "versicolor", 2: "virginica"}


@app.route("/")
def home():
    return "<h3>Iris Prediction API is Running</h3>"


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/metadata")
def metadata():
    return jsonify({
        "model_type": "RandomForestClassifier",
        "features": ["sepal length", "sepal width", "petal length", "petal width"],
        "target_classes": list(species.values())
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        if "features" not in data:
            return jsonify({"error": "Missing 'features' in request body"}), 400

        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)[0]
        return jsonify({
            "prediction": int(prediction),
            "species": species[int(prediction)]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


#  PE02 route
@app.route("/runtime")
def runtime():
    def safe_version(pkg):
        try:
            return pkg_version(pkg)
        except Exception:
            return None

    return jsonify({
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "hostname": socket.gethostname(),
        "packages": {
            "flask": safe_version("flask"),
            "scikit-learn": safe_version("scikit-learn"),
            "joblib": safe_version("joblib")
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
