"""SuperKart sales-forecast Streamlit frontend.

Talks to the Flask backend over the Docker network using the container name
"backend" (Docker's internal DNS) -- not localhost, not the public Codespace URL.
"""
import pandas as pd
import requests
import streamlit as st

BACKEND_URL = "http://backend:7860"

# Twin of REFERENCE_YEAR in SuperKart_Deployment.ipynb (Section 5, Store_Age_Years
# cell). Must stay byte-identical -- copied, not retyped. If these drift apart the
# model receives a feature it was never trained on with no error raised.
REFERENCE_YEAR = 2009

# Twin of PRODUCT_TYPE_CATEGORY_MAP in SuperKart_Deployment.ipynb (Section 5,
# Product_Type_Category cell). Must stay byte-identical -- copied, not retyped.
PRODUCT_TYPE_CATEGORY_MAP = {
    "Dairy": "Perishables",
    "Meat": "Perishables",
    "Seafood": "Perishables",
    "Fruits and Vegetables": "Perishables",
    "Breads": "Perishables",
    "Breakfast": "Perishables",
    "Frozen Foods": "Perishables",
    "Baking Goods": "Non-Perishables",
    "Canned": "Non-Perishables",
    "Health and Hygiene": "Non-Perishables",
    "Household": "Non-Perishables",
    "Snack Foods": "Non-Perishables",
    "Soft Drinks": "Non-Perishables",
    "Hard Drinks": "Non-Perishables",
    "Others": "Non-Perishables",
    "Starchy Foods": "Non-Perishables",
}

# Product_Id_char is the first two characters of Product_Id in the training data
# (instructions.md Section 7.3): FD = Food, DR = Drink, NC = Non-Consumable.
PRODUCT_ID_CHAR_OPTIONS = {
    "Food (FD)": "FD",
    "Drink (DR)": "DR",
    "Non-Consumable (NC)": "NC",
}

st.set_page_config(page_title="SuperKart Sales Forecast", layout="centered")
st.title("SuperKart Sales Forecast")

tab_online, tab_batch = st.tabs(["Online Prediction", "Batch Prediction"])

with tab_online:
    st.subheader("Predict revenue for a single product / store combination")

    col1, col2 = st.columns(2)
    with col1:
        product_weight = st.number_input("Product Weight", min_value=0.0, max_value=50.0, value=12.66, step=0.01)
        product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        product_allocated_area = st.slider("Product Allocated Area (fraction of store display area)", 0.0, 0.3, 0.05, step=0.001)
        product_mrp = st.number_input("Product MRP", min_value=0.0, max_value=500.0, value=147.0, step=0.01)
        product_id_char_label = st.selectbox("Product Category", list(PRODUCT_ID_CHAR_OPTIONS.keys()))

    with col2:
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Food Mart", "Supermarket Type1", "Supermarket Type2", "Departmental Store"])
        store_establishment_year = st.number_input(
            "Store Establishment Year", min_value=1950, max_value=REFERENCE_YEAR, value=2009, step=1,
        )
        product_type = st.selectbox("Product Type", sorted(PRODUCT_TYPE_CATEGORY_MAP.keys()))

    if st.button("Predict Sales"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar_content,
            "Product_Allocated_Area": product_allocated_area,
            "Product_MRP": product_mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_location_city_type,
            "Store_Type": store_type,
            "Product_Id_char": PRODUCT_ID_CHAR_OPTIONS[product_id_char_label],
            "Store_Age_Years": REFERENCE_YEAR - store_establishment_year,
            "Product_Type_Category": PRODUCT_TYPE_CATEGORY_MAP[product_type],
        }

        try:
            response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Product Store Sales Total: ${result['Product_Store_Sales_Total']:,.2f}")
            else:
                st.error(f"Backend returned status {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the backend: {e}")

with tab_batch:
    st.subheader("Predict revenue for a batch of products / stores")
    st.write(
        "Upload a CSV with the ten contract columns: Product_Weight, "
        "Product_Sugar_Content, Product_Allocated_Area, Product_MRP, Store_Size, "
        "Store_Location_City_Type, Store_Type, Product_Id_char, Store_Age_Years, "
        "Product_Type_Category."
    )
    uploaded_file = st.file_uploader("Upload batch CSV", type=["csv"])

    if uploaded_file is not None and st.button("Predict Batch"):
        try:
            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                timeout=60,
            )
            if response.status_code == 200:
                predictions = response.json()
                pred_df = pd.DataFrame(
                    {"Row": list(predictions.keys()), "Predicted_Product_Store_Sales_Total": list(predictions.values())}
                )
                st.dataframe(pred_df)
            else:
                st.error(f"Backend returned status {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the backend: {e}")
