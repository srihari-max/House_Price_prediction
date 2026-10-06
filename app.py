import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: 20px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    background-color: #e8f5e9;
    margin-top: 25px;
}

.result-title {
    font-size: 20px;
    color: #555;
}

.result-price {
    font-size: 38px;
    font-weight: 700;
    color: #16803c;
}

.info-box {
    padding: 18px;
    border-radius: 12px;
    background-color: #eef3ff;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = joblib.load("house_price_model.pkl")

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    '<div class="title">🏠 House Price Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Property Price Estimator</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------------------------
# PROPERTY DETAILS
# ---------------------------------------------------

st.markdown(
    '<div class="section-title">🏡 Property Details</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    area = st.number_input(
        "Area (sq.ft)",
        min_value=100,
        value=1500,
        step=100
    )

with col2:
    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        value=3,
        step=1
    )

with col3:
    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        value=2,
        step=1
    )

col1, col2, col3 = st.columns(3)

with col1:
    stories = st.number_input(
        "Stories",
        min_value=1,
        value=2,
        step=1
    )

with col2:
    parking = st.number_input(
        "Parking Spaces",
        min_value=0,
        value=1,
        step=1
    )

with col3:
    mainroad = st.selectbox(
        "Main Road",
        ["yes", "no"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    guestroom = st.selectbox(
        "Guest Room",
        ["yes", "no"]
    )

with col2:
    basement = st.selectbox(
        "Basement",
        ["yes", "no"]
    )

with col3:
    hotwaterheating = st.selectbox(
        "Hot Water Heating",
        ["yes", "no"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    airconditioning = st.selectbox(
        "Air Conditioning",
        ["yes", "no"]
    )

with col2:
    prefarea = st.selectbox(
        "Preferred Area",
        ["yes", "no"]
    )

with col3:
    furnishingstatus = st.selectbox(
        "Furnishing Status",
        [
            "furnished",
            "semi-furnished",
            "unfurnished"
        ]
    )

st.write("")

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

predict_button = st.button(
    "🔮 Predict House Price",
    use_container_width=True
)

if predict_button:

    new_house = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "stories": [stories],
        "mainroad": [mainroad],
        "guestroom": [guestroom],
        "basement": [basement],
        "hotwaterheating": [hotwaterheating],
        "airconditioning": [airconditioning],
        "parking": [parking],
        "prefarea": [prefarea],
        "furnishingstatus": [furnishingstatus]
    })

    predicted_price = model.predict(new_house)[0]

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-title">
                Estimated House Price
            </div>
            <div class="result-price">
                ₹{predicted_price:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-title">🤖 About the Model</div>',
    unsafe_allow_html=True
)

info1, info2, info3 = st.columns(3)

with info1:
    st.metric(
        "Model",
        "Gradient Boosting"
    )

with info2:
    st.metric(
        "Features",
        "12"
    )

with info3:
    st.metric(
        "R² Score",
        "0.666"
    )

st.markdown(
    """
    <div class="info-box">
    <b>How it works:</b><br>
    The user enters property details such as area, bedrooms,
    bathrooms, location-related features, parking and furnishing
    status. The trained machine learning model processes these
    features and estimates the house price.
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")
st.caption("House Price Prediction • Machine Learning Project")