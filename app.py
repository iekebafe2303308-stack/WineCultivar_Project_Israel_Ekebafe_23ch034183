from pathlib import Path
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = Path("model") / "wine_cultivar_model.pkl"

# Load trained model (pipeline with scaler + classifier)
model = joblib.load(MODEL_PATH)

# Feature order must match training
FEATURES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "flavanoids",
    "proline",
]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    if request.method == "POST":
        try:
            values = [float(request.form.get(feat, "")) for feat in FEATURES]
            if any(v is None for v in values):
                raise ValueError("All fields are required.")

            # Model expects 2D array
            pred = model.predict([values])[0]
            # Convert to 1, 2, 3 (dataset classes are 0,1,2)
            prediction = int(pred) + 1
        except Exception as exc:  # noqa: BLE001
            error = f"Invalid input: {exc}"

    return render_template(
        "index.html",
        features=FEATURES,
        prediction=prediction,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
