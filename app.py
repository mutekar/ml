from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
import json

app = Flask(__name__)

# Load the model, scaler, and country mapping
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
with open("country_mapping.json", "r") as f:
    country_mapping = json.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get user input
            country = request.form["country"]
            revenue_ttm = float(request.form["revenue_ttm"])
            earnings_ttm = float(request.form["earnings_ttm"])
            marketcap = float(request.form["marketcap"])
            pe_ratio_ttm = float(request.form["pe_ratio_ttm"])
            dividend_yield_ttm = float(request.form["dividend_yield_ttm"])

            # Transform country name to its encoded value
            country_encoded = country_mapping.get(country, 0)

            # Prepare input data
            input_data = np.array([[country_encoded, revenue_ttm, earnings_ttm, marketcap, pe_ratio_ttm, dividend_yield_ttm]])
            input_data_scaled = scaler.transform(input_data)

            # Predict
            prediction_log = model.predict(input_data_scaled)
            prediction_price = np.expm1(prediction_log)[0]

            return render_template("index.html", result=f"Predicted Price (GBP): {prediction_price:.2f}")
        except Exception as e:
            return render_template("index.html", result=f"Error: {str(e)}")

    return render_template("index.html", result="")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
