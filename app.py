import os, pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mymodel.pkl")
model = None
try:
    # Only load pickle files from sources you trust.
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as exc:
    app.logger.error("Could not load model: %s", exc)

FEATURES = ["Glucose", "BloodPressure", "Insulin", "BMI", "Age"]

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/api/health")
def health():
    return jsonify({"status": "ok" if model is not None else "model_unavailable",
                    "model_loaded": model is not None}), (200 if model is not None else 503)

@app.post("/api/predict")
def predict():
    if model is None:
        return jsonify({"error": "Model unavailable. Check Render logs and install the model's required libraries."}), 503
    data = request.get_json(silent=True) or {}
    values = []
    for feature in FEATURES:
        try:
            value = float(data.get(feature, ""))
        except (TypeError, ValueError):
            return jsonify({"error": f"Enter a valid number for {feature}."}), 400
        if not np.isfinite(value) or value < 0:
            return jsonify({"error": f"{feature} must be a non-negative number."}), 400
        values.append(value)
    try:
        row = np.array([values], dtype=float)
        prediction = int(np.asarray(model.predict(row)).reshape(-1)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(row)
                classes = list(getattr(model, "classes_", []))
                idx = classes.index(1) if 1 in classes else (1 if probs.shape[1] > 1 else 0)
                probability = float(probs[0][idx])
            except Exception:
                probability = None
        return jsonify({"prediction": prediction,
                        "probability": probability,
                        "disclaimer": "This is a model output, not a medical diagnosis."})
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check that the model's training features and preprocessing match this API."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
