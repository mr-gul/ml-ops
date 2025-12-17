from flask import Flask, request, jsonify
import joblib
from pathlib import Path

app = Flask(__name__)
MODEL_PATH = Path("artifacts/model.pkl")

# Load model
model = joblib.load(MODEL_PATH)

# Define class labels
TARGETS = ["Setosa", "Versicolor", "Virginica"]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data["features"]
    
    # Predict the class index
    pred_index = model.predict([features])[0]
    
    # Map index to target name
    pred_class = TARGETS[pred_index]
    
    # Return only the predicted class
    return jsonify({"prediction": pred_class})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
