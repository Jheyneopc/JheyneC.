from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model/iris_model.pkl")

# label : species
species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}

@app.route("/")
def index():
    return "MLOps Flask API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    features = np.array(data["features"]).reshape(1, -1)

    pred_class = int(model.predict(features)[0])
    pred_species = species_map[pred_class]

    return jsonify({
        "prediction": pred_class,
        "species": pred_species
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
