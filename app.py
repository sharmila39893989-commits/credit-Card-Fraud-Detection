import streamlit as st
import pandas as pd
import pickle

# -------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -------------------------------------------------
# 2. CUSTOM CSS - NEW COLOURFUL THEME
# -------------------------------------------------

st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(
        135deg,
        #ecfccb 0%,
        #ccfbf1 35%,
        #ffedd5 70%,
        #fef9c3 100%
    );
}

/* Main Title */
.main-title {
    text-align: center;
    font-size: 44px;
    font-weight: bold;
    color: #431407;
    padding: 20px;
    text-shadow: 2px 2px 3px #fde68a;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 19px;
    font-weight: bold;
    color: #164e63;
    margin-bottom: 25px;
}

/* Section Heading */
.section-title {
    color: #4c0519;
    background: linear-gradient(
        90deg,
        #fde68a,
        #a7f3d0,
        #bae6fd,
        #fed7aa
    );
    font-size: 26px;
    font-weight: bold;
    text-align: center;
    padding: 14px;
    border-radius: 20px;
    border: 3px solid #0f766e;
    box-shadow: 0px 5px 12px rgba(15, 118, 110, 0.25);
    margin-top: 25px;
    margin-bottom: 18px;
}

/* Information Card */
.info-card {
    background: linear-gradient(
        135deg,
        #fef3c7,
        #cffafe,
        #d9f99d
    );
    padding: 25px;
    border-radius: 22px;
    border: 3px solid #a16207;
    box-shadow: 0px 6px 15px rgba(120, 53, 15, 0.20);
    margin-bottom: 20px;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        #fef08a,
        #a7f3d0,
        #bae6fd
    );
    padding: 18px;
    border-radius: 18px;
    border: 3px solid #0e7490;
    box-shadow: 0px 5px 12px rgba(14, 116, 144, 0.20);
}

/* Metric Text */
[data-testid="stMetricLabel"] {
    color: #431407 !important;
    font-weight: bold !important;
}

[data-testid="stMetricValue"] {
    color: #164e63 !important;
    font-weight: bold !important;
}

/* Input Area */
[data-testid="stNumberInput"] {
    background: linear-gradient(
        135deg,
        #cffafe,
        #ecfccb
    );
    padding: 12px;
    border-radius: 16px;
    border: 3px solid #0d9488;
    box-shadow: 0px 4px 10px rgba(13, 148, 136, 0.20);
    margin-bottom: 12px;
}

/* Input Labels */
label {
    color: #4c0519 !important;
    font-weight: bold !important;
    font-size: 15px !important;
}

/* Input Text */
input {
    color: #431407 !important;
    background-color: #fefce8 !important;
    font-weight: bold !important;
}

/* Input Border */
div[data-baseweb="input"] {
    border: 2px solid #a16207;
    border-radius: 12px;
}

/* Predict Button */
div.stButton > button {
    background: linear-gradient(
        90deg,
        #854d0e,
        #0f766e,
        #0369a1,
        #65a30d
    );
    color: #ffffff !important;
    border: 4px solid #fef08a;
    border-radius: 18px;
    padding: 15px;
    font-size: 20px;
    font-weight: bold;
    width: 100%;
    box-shadow: 0px 6px 15px rgba(67, 20, 7, 0.30);
}

/* Button Hover */
div.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #be123c,
        #7e22ce,
        #c2410c,
        #047857
    );
    color: #ffffff !important;
    border: 4px solid #67e8f9;
}

/* Information Message */
div[data-testid="stAlert"] {
    background: linear-gradient(
        90deg,
        #fef3c7,
        #cffafe,
        #d9f99d
    );
    color: #431407;
    border: 3px solid #a16207;
    border-radius: 16px;
    font-weight: bold;
}

/* Success Result */
div[data-testid="stAlert"][kind="success"] {
    background: linear-gradient(
        90deg,
        #d9f99d,
        #a7f3d0,
        #99f6e4
    );
    border: 3px solid #15803d;
}

/* Error Result */
div[data-testid="stAlert"][kind="error"] {
    background: linear-gradient(
        90deg,
        #fed7aa,
        #fde68a,
        #fecaca
    );
    border: 3px solid #c2410c;
}

/* Progress Bar */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(
        90deg,
        #0f766e,
        #65a30d,
        #eab308
    );
}

/* Divider */
hr {
    border: 2px solid #a16207;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# 3. LOAD MODEL AND SCALER
# -------------------------------------------------

try:

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

except FileNotFoundError:

    st.error(
        "Model files not found! "
        "Please run the notebook and save "
        "model.pkl and scaler.pkl first."
    )

    st.stop()

# -------------------------------------------------
# 4. HEADER
# -------------------------------------------------

st.markdown(
    '<div class="main-title">'
    '💳 Credit Card Fraud Detection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    '🌈 AI-Powered Transaction Classification System 🌈'
    '</div>',
    unsafe_allow_html=True
)

# -------------------------------------------------
# 5. INFORMATION CARDS
# -------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🤖 Model",
        type(model).__name__
    )

with col2:
    st.metric(
        "📊 Task",
        "Classification"
    )

with col3:
    st.metric(
        "🎯 Classes",
        "Normal / Fraud"
    )

st.divider()

# -------------------------------------------------
# 6. TRANSACTION DETAILS
# -------------------------------------------------

st.markdown(
    '<div class="section-title">'
    '📝 Transaction Details'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="info-card">',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    time = st.number_input(
        "⏰ Transaction Time",
        min_value=0.0,
        value=0.0,
        step=1.0,
        key="time"
    )

with col2:

    amount = st.number_input(
        "💰 Transaction Amount",
        min_value=0.0,
        value=100.0,
        step=0.01,
        key="amount"
    )

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# -------------------------------------------------
# 7. V1 TO V28 FEATURES
# -------------------------------------------------

st.markdown(
    '<div class="section-title">'
    '🔢 Transaction Features'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="info-card">',
    unsafe_allow_html=True
)

features = []

columns = st.columns(4)

for i in range(1, 29):

    feature = f"V{i}"

    with columns[(i - 1) % 4]:

        value = st.number_input(
            f"✨ Feature {i}",
            value=0.0,
            format="%.6f",
            key=f"feature_{i}"
        )

        features.append(value)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# -------------------------------------------------
# 8. PREDICTION BUTTON
# -------------------------------------------------

if st.button("🔍 Predict Transaction"):

    input_data = pd.DataFrame(
        [[time] + features + [amount]],
        columns=[
            "Time"
        ] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    )

    try:

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '📋 Prediction Result'
            '</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:

            st.error(
                "⚠️ Fraudulent Transaction Detected"
            )

        else:

            st.success(
                "✅ Normal Transaction"
            )

        st.info(
            f"Predicted Class: {prediction}"
        )

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(
                input_scaled
            )[0][1]

            st.metric(
                "Fraud Probability",
                f"{probability:.2%}"
            )

            st.progress(float(probability))

        else:

            st.warning(
                "Probability is not available "
                "for this model."
            )

        st.caption(
            "This prediction is for educational purposes only."
        )

    except Exception as error:

        st.error(
            f"Prediction error: {error}"
        )