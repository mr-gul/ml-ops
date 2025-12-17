import json
from pathlib import Path
import numpy as np
import argparse 
import joblib

MODEL_PATH=Path("artifacts/model.pkl")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model
def main():
    parser = argparse.ArgumentParser(description="Make predictions using the trained model.")
    parser.add_argument("--input", required=True, help="Provide feature as list string. E.g., '[1.0, 2.0, 3.0]'")
    args = parser.parse_args()
    
    try:
        features = json.loads(args.input)
    except json.JSONDecodeError:
        raise ValueError("Input features must be a valid JSON list string.")
    
    X = np.array(features).reshape(1, -1)
    model = load_model()
    prediction = model.predict(X)
    
    target_names = ["Setosa", "Versicolor", "Virginica"]
    print(f"Prediction: {target_names[prediction[0]]}")
if __name__ == "__main__":
    main()