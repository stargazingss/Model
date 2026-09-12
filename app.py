import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Paddy Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "page" not in st.session_state:
    st.session_state.page = "welcome"

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "robust_standard_scaler.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
columns = joblib.load(COLUMNS_PATH)

st.markdown(
    """
    <style>

    .stApp {
        background-color: #F8F5EA;
    }

    .block-container {
        max-width: 1120px !important;
        padding-top: 60px !important;
        padding-bottom: 50px !important;
        padding-left: 32px !important;
        padding-right: 32px !important;
    }

    .hero-title {
        color: #596A32 !important;
        font-size: 44px !important;
        font-weight: 800 !important;
        line-height: 1.1 !important;
        letter-spacing: -1px !important;
        margin-top: 0 !important;
        margin-bottom: 8px !important;
    }

    .hero-label {
        color: #8A741F !important;
        font-size: 11px !important;
        font-weight: 800 !important;
        letter-spacing: 1.7px !important;
        margin-bottom: 7px !important;
    }

    .hero-description {
        color: #777264 !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        max-width: 760px !important;
        margin-top: 0 !important;
    }

    .model-badge {
        background-color: #FFFDF7 !important;
        border: 1px solid #D8CC91 !important;
        border-radius: 999px !important;
        color: #596A32 !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        padding: 8px 14px !important;
        white-space: nowrap !important;
        text-align: center !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid #C9B84A !important;
        border-radius: 17px !important;
        box-shadow: 0 4px 14px rgba(70, 65, 40, 0.05) !important;
        padding: 22px !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #FFFFFF !important;
        border-radius: 17px !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] h2,
    div[data-testid="stVerticalBlockBorderWrapper"] h3,
    h2,
    h3 {
        color: #596A32 !important;
        font-weight: 800 !important;
        margin-top: 0 !important;
    }

    h2 a,
    h3 a,
    h2 a svg,
    h3 a svg {
        display: none !important;
    }

    .stCaption {
        color: #888477 !important;
        font-size: 12px !important;
    }

    label {
        color: #414638 !important;
        font-size: 14px !important;
        font-weight: 700 !important;
    }

    div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D8D3C4 !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="input"] input {
        background-color: #FFFFFF !important;
        color: #333333 !important;
        font-size: 14px !important;
    }

    div[data-baseweb="input"]:focus-within {
        border: 1.5px solid #C9B84A !important;
        box-shadow: 0 0 0 2px rgba(201, 184, 74, 0.10) !important;
    }

    div[data-testid="stForm"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
    }

    div[data-testid="stFormSubmitButton"] button {
        background-color: #596A32 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        min-height: 48px !important;
        font-size: 14px !important;
        font-weight: 700 !important;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #4E5D2C !important;
    }

    .welcome-button button {
        background-color: #596A32 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        min-height: 50px !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        transition: all 0.2s ease !important;
    }

    .welcome-button button:hover {
        background-color: #4E5D2C !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid #D8CC91 !important;
        border-radius: 16px !important;
        padding: 24px !important;
        min-height: 145px !important;
        box-shadow: 0 4px 14px rgba(70, 65, 40, 0.05) !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #777264 !important;
        font-size: 13px !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #596A32 !important;
        font-size: 30px !important;
        font-weight: 800 !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    .welcome-page {
        min-height: 62vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }

    .welcome-label {
        color: #8A741F;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.8px;
        margin-bottom: 16px;
    }

    .welcome-title {
        color: #596A32;
        font-size: 52px;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -1.5px;
        margin: 0 0 18px 0;
    }

    .welcome-description {
        color: #777264;
        font-size: 16px;
        line-height: 1.7;
        max-width: 650px;
        margin: 0;
    }

    .welcome-button {
        margin-top: 12px;
    }

    .footer-text {
        text-align: center !important;
        color: #999488 !important;
        font-size: 11px !important;
        width: 100% !important;
        margin-top: 32px !important;
    }

    @media (max-width: 768px) {

        .block-container {
            padding-left: 18px !important;
            padding-right: 18px !important;
            padding-top: 42px !important;
        }

        .hero-title {
            font-size: 34px !important;
        }

        .hero-description {
            font-size: 14px !important;
        }

        .welcome-title {
            font-size: 38px !important;
        }

        .welcome-description {
            font-size: 14px !important;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


if st.session_state.page == "welcome":

    st.markdown(
        '<div class="welcome-page">'
        '<div class="welcome-label">RICE YIELD ESTIMATION</div>'
        '<div class="welcome-title">Paddy Yield Prediction</div>'
        '<div class="welcome-description">'
        'Estimate your expected paddy yield using field conditions, '
        'seed usage, fertilization, nursery preparation, and crop protection data.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1.2, 1, 1.2])

    with col2:

        st.markdown(
            '<div class="welcome-button">',
            unsafe_allow_html=True
        )

        if st.button(
            "Start Yield Prediction →",
            width="stretch"
        ):
            st.session_state.page = "input"
            st.rerun()

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <p class="footer-text">
            Paddy Yield Prediction System
        </p>
        """,
        unsafe_allow_html=True
    )

    st.stop()


header_left, header_right = st.columns(
    [4.5, 1.5],
    vertical_alignment="top"
)

with header_left:

    st.caption("RICE YIELD ESTIMATION")

    st.markdown(
        '<p class="hero-title">Paddy Yield Prediction</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p class="hero-description">
        Estimate expected rice yield in kilograms based on field size,
        seed usage, fertilization, nursery preparation, and crop protection inputs.
        </p>
        """,
        unsafe_allow_html=True
    )

with header_right:

    st.write("")

    st.markdown(
        '<p class="model-badge">Optimized Random Forest Model</p>',
        unsafe_allow_html=True
    )


st.markdown(
    """
    <hr style="
        border: none;
        border-top: 1px solid #D5D1C6;
        margin: 28px 0 24px 0;
    ">
    """,
    unsafe_allow_html=True
)


with st.form("prediction_form"):

    with st.container(border=True):

        st.subheader("Field Conditions")

        st.caption(
            "Basic information about the rice-growing area."
        )

        col1, col2 = st.columns(2, gap="large")

        with col1:

            hectares = st.number_input(
                "Hectares",
                min_value=0.01,
                value=2.5,
                step=0.1
            )

            st.caption(
                "Total active rice-growing area currently planted with rice."
            )

        with col2:

            seedrate = st.number_input(
                "Seed Rate (kg)",
                min_value=0.0,
                value=150.0,
                step=1.0
            )

            st.caption(
                "Total amount of rice seed used for the planting area."
            )

    st.markdown(
        """
        <hr style="
            border: none;
            border-top: 1px solid #D5D1C6;
            margin: 26px 0;
        ">
        """,
        unsafe_allow_html=True
    )

    with st.container(border=True):

        st.subheader("Nursery")

        st.caption(
            "Information related to nursery area and preparation."
        )

        col1, col2 = st.columns(2, gap="large")

        with col1:

            nursery_area = st.number_input(
                "Nursery Area (Cents)",
                min_value=0.0,
                value=120.0,
                step=1.0
            )

            st.caption(
                "Area allocated for preparing rice seedlings before transplantation."
            )

        with col2:

            lp_nursery = st.number_input(
                "LP Nursery Area (Tonnes)",
                min_value=0.0,
                value=6.0,
                step=0.1
            )

            st.caption(
                "Land preparation input used for the nursery area."
            )

    st.markdown(
        """
        <hr style="
            border: none;
            border-top: 1px solid #D5D1C6;
            margin: 26px 0;
        ">
        """,
        unsafe_allow_html=True
    )

    with st.container(border=True):

        st.subheader("Field Preparation")

        st.caption(
            "Inputs used during preparation of the main rice field."
        )

        col1, col2 = st.columns(2, gap="large")

        with col1:

            lp_mainfield = st.number_input(
                "LP Main Field (Tonnes)",
                min_value=0.0,
                value=75.0,
                step=1.0
            )

            st.caption(
                "Land preparation input applied to the main rice-growing field."
            )

        with col2:

            trash = st.number_input(
                "Trash (Bundles)",
                min_value=0.0,
                value=540.0,
                step=1.0
            )

            st.caption(
                "Amount of agricultural residue or plant material present in the field."
            )

    st.markdown(
        """
        <hr style="
            border: none;
            border-top: 1px solid #D5D1C6;
            margin: 26px 0;
        ">
        """,
        unsafe_allow_html=True
    )

    with st.container(border=True):

        st.subheader("Fertilization")

        st.caption(
            "Fertilizer inputs applied during rice cultivation."
        )

        col1, col2 = st.columns(2, gap="large")

        with col1:

            dap = st.number_input(
                "DAP (kg)",
                min_value=0.0,
                value=240.0,
                step=1.0
            )

            st.caption(
                "Amount of DAP fertilizer applied to the rice field."
            )

            urea = st.number_input(
                "Urea (kg)",
                min_value=0.0,
                value=162.78,
                step=1.0
            )

            st.caption(
                "Amount of urea fertilizer applied during cultivation."
            )

        with col2:

            potash = st.number_input(
                "Potash (kg)",
                min_value=0.0,
                value=62.28,
                step=1.0
            )

            st.caption(
                "Amount of potash fertilizer applied to the rice field."
            )

            micronutrients = st.number_input(
                "Micronutrients (kg)",
                min_value=0.0,
                value=90.0,
                step=1.0
            )

            st.caption(
                "Amount of micronutrient fertilizer applied during cultivation."
            )

    st.markdown(
        """
        <hr style="
            border: none;
            border-top: 1px solid #D5D1C6;
            margin: 26px 0;
        ">
        """,
        unsafe_allow_html=True
    )

    with st.container(border=True):

        st.subheader("Crop Protection")

        st.caption(
            "Inputs related to weed and pest management."
        )

        col1, col2 = st.columns(2, gap="large")

        with col1:

            weed = st.number_input(
                "Weed Control",
                min_value=0.0,
                value=12.0,
                step=1.0
            )

            st.caption(
                "Amount of weed-control treatment used during rice cultivation."
            )

        with col2:

            pest = st.number_input(
                "Pest Control (ml)",
                min_value=0.0,
                value=3600.0,
                step=100.0
            )

            st.caption(
                "Amount of pest-control treatment used during cultivation."
            )

    st.write("")

    predict_button = st.form_submit_button(
        "Predict Paddy Yield",
        width="stretch"
    )


if predict_button:

    try:

        input_data = {
            "Hectares ": hectares,
            "Micronutrients_70Days": micronutrients,
            "Potassh_50Days": potash,
            "Urea_40Days": urea,
            "Pest_60Day(in ml)": pest,
            "LP_Mainfield(in Tonnes)": lp_mainfield,
            "DAP_20days": dap,
            "Trash(in bundles)": trash,
            "Seedrate(in Kg)": seedrate,
            "LP_nurseryarea(in Tonnes)": lp_nursery,
            "Weed28D_thiobencarb": weed,
            "Nursery area (Cents)": nursery_area
        }

        input_df = pd.DataFrame([input_data])

        input_df = input_df[columns]

        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)[0]

        yield_per_hectare = prediction / hectares

        st.markdown(
            """
            <hr style="
                border: none;
                border-top: 1px solid #D5D1C6;
                margin: 28px 0 24px 0;
            ">
            """,
            unsafe_allow_html=True
        )

        st.subheader("Prediction Result")

        st.caption(
            "Estimated production based on the provided cultivation inputs."
        )

        result_col1, result_col2 = st.columns(
            2,
            gap="large"
        )

        with result_col1:

            st.metric(
                "Estimated Total Yield",
                f"{prediction:,.2f} kg"
            )

        with result_col2:

            st.metric(
                "Yield per Hectare",
                f"{yield_per_hectare:,.2f} kg/ha"
            )

        if yield_per_hectare < 4000:

            productivity_status = "Low"

        elif yield_per_hectare <= 6000:

            productivity_status = "Moderate"

        else:

            productivity_status = "High"

        st.write("")

        st.info(
            f"Productivity Status: {productivity_status}"
        )

        st.markdown(
            """
            <hr style="
                border: none;
                border-top: 1px solid #D5D1C6;
                margin: 28px 0 24px 0;
            ">
            """,
            unsafe_allow_html=True
        )

        st.subheader("Input Recommendations")

        recommendations = []

        seed_per_hectare = seedrate / hectares

        if seed_per_hectare < 20:

            recommendations.append(
                (
                    "Seed Rate",
                    "Seed usage per hectare is relatively low. "
                    "Review the planting density based on the cultivation area.",
                    "warning"
                )
            )

        elif seed_per_hectare > 60:

            recommendations.append(
                (
                    "Seed Rate",
                    "Seed usage per hectare is relatively high. "
                    "Consider reviewing the seed rate used for the field.",
                    "warning"
                )
            )

        else:

            recommendations.append(
                (
                    "Seed Rate",
                    "Seed usage per hectare is within the current reference range.",
                    "success"
                )
            )

        lp_per_hectare = lp_mainfield / hectares

        if lp_per_hectare < 20:

            recommendations.append(
                (
                    "Field Preparation",
                    "Main-field preparation input per hectare is relatively low. "
                    "Review whether the field preparation is sufficient.",
                    "warning"
                )
            )

        else:

            recommendations.append(
                (
                    "Field Preparation",
                    "Main-field preparation input is within the current reference range.",
                    "success"
                )
            )

        fertilizer_messages = []

        if urea < 150:

            fertilizer_messages.append(
                "Urea input is relatively low."
            )

        if potash < 50:

            fertilizer_messages.append(
                "Potash input is relatively low."
            )

        if dap < 200:

            fertilizer_messages.append(
                "DAP input is relatively low."
            )

        if micronutrients > 70:

            fertilizer_messages.append(
                "Micronutrient input is relatively high."
            )

        if fertilizer_messages:

            recommendations.append(
                (
                    "Fertilization",
                    " ".join(fertilizer_messages)
                    + " Review fertilizer application according to field requirements.",
                    "warning"
                )
            )

        else:

            recommendations.append(
                (
                    "Fertilization",
                    "Fertilizer inputs are within the current reference ranges.",
                    "success"
                )
            )

        crop_messages = []

        if weed < 10:

            crop_messages.append(
                "Weed-control input is relatively low."
            )

        if pest < 2500:

            crop_messages.append(
                "Pest-control input is relatively low."
            )

        if crop_messages:

            recommendations.append(
                (
                    "Crop Protection",
                    " ".join(crop_messages)
                    + " Review crop-protection requirements based on field conditions.",
                    "info"
                )
            )

        else:

            recommendations.append(
                (
                    "Crop Protection",
                    "Crop-protection inputs are within the current reference ranges.",
                    "success"
                )
            )

        for category, message, status in recommendations:

            if status == "warning":

                st.warning(
                    f"{category}\n\n{message}"
                )

            elif status == "info":

                st.info(
                    f"{category}\n\n{message}"
                )

            else:

                st.success(
                    f"{category}\n\n{message}"
                )

        st.markdown(
            """
            <hr style="
                border: none;
                border-top: 1px solid #D5D1C6;
                margin: 28px 0 24px 0;
            ">
            """,
            unsafe_allow_html=True
        )

        st.subheader("Model Performance")

        st.caption(
            "Performance of the optimized Random Forest model on the test data."
        )

        performance_col1, performance_col2, performance_col3 = st.columns(
            3,
            gap="large"
        )

        with performance_col1:

            st.metric(
                "R² Score",
                "0.9903"
            )

        with performance_col2:

            st.metric(
                "MAE",
                "657.62 kg"
            )

        with performance_col3:

            st.metric(
                "RMSE",
                "912.78 kg"
            )

        st.caption(
            "Note: This prediction is generated by a machine learning model "
            "based on historical agricultural data. The result is an estimation "
            "and not a guaranteed harvest outcome."
        )

        st.markdown(
            """
            <hr style="
                border: none;
                border-top: 1px solid #D5D1C6;
                margin: 28px 0 24px 0;
            ">
            """,
            unsafe_allow_html=True
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {str(e)}"
        )


st.markdown(
    """
    <p class="footer-text">
        Paddy Yield Prediction System
    </p>
    """,
    unsafe_allow_html=True
)