"""SuperKart sales-forecast API.

Endpoint paths (/v1/predict, /v1/predictbatch) and the app object name
(superkart_api) are fixed contracts -- the notebook's inference cells (Section 10)
and the Dockerfile CMD both depend on them exactly as written here.
"""
import pandas as pd
from flask import Flask, jsonify, request
import joblib

superkart_api = Flask(__name__)

# Loaded once at module level, not per request.
model = joblib.load("superkart_model.joblib")

# Must match instructions.md Section 7.1 exactly -- the order does not matter to
# the pipeline (it selects by column name), but the set must be exact.
FEATURE_LIST = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]


@superkart_api.get("/")
def health_check():
    return "SuperKart Sales Forecast API is running."


@superkart_api.post("/v1/predict")
def predict():
    try:
        payload = request.get_json(force=True)
        missing = [f for f in FEATURE_LIST if f not in payload]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400

        row = pd.DataFrame([{f: payload[f] for f in FEATURE_LIST}])
        prediction = model.predict(row)[0]

        # NumPy float32/float64 is not JSON-serializable -- cast to native Python float.
        return jsonify({"Product_Store_Sales_Total": round(float(prediction), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@superkart_api.post("/v1/predictbatch")
def predict_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided under form key \'file\'"}), 400

        batch_df = pd.read_csv(request.files["file"])
        missing = [f for f in FEATURE_LIST if f not in batch_df.columns]
        if missing:
            return jsonify({"error": f"Missing required columns: {missing}"}), 400

        predictions = model.predict(batch_df[FEATURE_LIST])
        result = {
            str(idx): round(float(pred), 2)
            for idx, pred in zip(batch_df.index, predictions)
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860)
