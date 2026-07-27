from flask import Flask, request, jsonify
import pandas as pd

from src.predict import predict_loan

app = Flask(__name__)


@app.route("/")
def home():
    return {
        "application": "Loan Default Prediction API",
        "version": "1.0.0",
        "status": "Running"
    }


@app.route("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    input_df = pd.DataFrame([data])

    prediction, probability = predict_loan(input_df)

    return jsonify({
        "prediction": prediction,
        "probability": round(float(probability), 4)
    })


if __name__ == "__main__":
    app.run(debug=True)