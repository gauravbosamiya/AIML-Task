import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000"  

st.set_page_config(
    page_title="Design Demand Predictor",
    layout="centered"
)

@st.cache_data(ttl=30)
def get_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json()
    except:
        return None

health = get_health()

st.title("👗 Design Demand Predictor")
st.markdown("Upload a design image and enter the rate to predict expected order quantity.")

if health:
    st.success(f"API Connected ")
else:
    st.error("API Offline — make sure FastAPI server is running.")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload Design Image",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Design", width='stretch')

with col2:
    rate_max_val = health["rate_max"] if health else 10000.0

    rate = st.number_input(
        "Rate",
        min_value=1.0,
        max_value=100000.0,
        value=1000.0,
        step=50.0,
        help=f"Training rate max was ₹{rate_max_val:.0f}"
    )

    if health and rate > health["rate_max"]:
        st.warning(f"Rate exceeds training max ₹{health['rate_max']:.0f}")

    st.markdown("###")
    predict_btn = st.button("Predict Demand", width='stretch', type="primary")

st.divider()

if predict_btn:
    if uploaded_file is None:
        st.error("Please upload a design image first.")
    elif not health:
        st.error("API is offline. Cannot make predictions.")
    else:
        with st.spinner("Analyzing design..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    files={"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    data={"rate": rate},
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    st.success("Prediction Complete!")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Predicted Qty",  f"{result['predicted_qty']} units")
                    m2.metric("Rate",           f"₹{result['rate']:.0f}")
                    m3.metric("Est. Revenue",   f"₹{result['estimated_revenue']:,}")

                    level = result["demand_level"]
                    if level == "low":
                        st.info("Low demand expected (< 20 units)")
                    elif level == "moderate":
                        st.warning("Moderate demand expected (20–60 units)")
                    else:
                        st.success("High demand expected (60+ units)")

                    if result["rate_warning"]:
                        st.warning("Rate is higher than training data — prediction may be less accurate.")

                else:
                    st.error(f"API Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Is the FastAPI server running?")
            except requests.exceptions.Timeout:
                st.error("Request timed out. Try again.")