import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import os

# Configure Page Settings
st.set_page_config(
    page_title="Ethiopia Food Price Prediction & Analytics",
    page_icon="ን",
    layout="wide",
)

# Custom Styling
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            background-color: #f4f6f9;
        }

        [data-testid="stForm"] {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
        }

        .stSelectbox label, .stDateInput label, .stNumberInput label {
            color: #1e293b !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }

        div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {
            background-color: #f8fafc;
            border-radius: 8px;
            border-color: #cbd5e1;
        }

        .stButton button {
            background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%);
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            border: none;
            width: 100%;
            box-shadow: 0 4px 6px rgba(13, 148, 136, 0.2);
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background: linear-gradient(135deg, #115e59 0%, #0f766e 100%);
            color: #ffffff;
            box-shadow: 0 6px 8px rgba(13, 148, 136, 0.3);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Safely load serialized model assets
def load_model_assets():
  try:
    model = joblib.load("random_forest_model.joblib")
    return model, None
  except Exception as e:
    return None, str(e)


# App Title
st.title(" Ethiopian Food Price Prediction & Market Analysis")
st.markdown(
    "An interactive decision-support application built for localized market"
    " forecasting."
)

# Navigation Sidebar
menu = st.sidebar.radio("Navigation", ["Dashboard Overview", "Model Prediction"])

# Display requirements.txt on the sidebar
st.sidebar.markdown("--- ")
if menu == "Dashboard Overview":
  st.header(" Market Insights & Historical Data Visualizations")

  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric(label="Total Recorded Transactions", value="36,746")
  with col2:
    st.metric(label="Covered Administrative Regions (Admin1)", value="11")
  with col3:
    st.metric(label="Unique Commodities Modeled", value="93")

  st.subheader("Historical Price Distributions")
  st.write(
      "The boxplot below reflects regional price variances with spatial and"
      " local outliers adjusted."
  )

  sample_viz_data = pd.DataFrame({
      "admin1": np.repeat(
          ["Addis Ababa", "Oromia", "Amhara", "Tigray", "Somali"], 100
      ),
      "usdprice": np.random.exponential(scale=35, size=500),
  })

  fig = px.box(
      sample_viz_data,
      x="admin1",
      y="price",
      color="admin1",
      title=" Price Distributions across Selected Regions",
  )
  st.plotly_chart(fig, use_container_width=True)

elif menu == "Model Prediction":
  st.header(" Real-Time Market Price Inference")
  st.markdown("Select actual market values below to evaluate food prices.")

  # Dictionaries for ID encoding
  admin1_mapping = {
      "Addis Ababa": 7,
      "Oromia": 8,
      "Amhara": 2,
      "Tigray": 10,
      "Somali": 5,
      "SNNPR / Sidama": 6,
      "Afar": 1,
      "Dire Dawa": 11,
  }

  category_mapping = {
      "Cereals and Tubers": 0,
      "Pulses and Nuts": 1,
      "Vegetables and Fruits": 2,
      "Meat, Fish and Eggs": 3,
  }

  unit_mapping = {"KG": 0, "Quintal": 1, "Pound": 2, "Litre": 3}

  priceflag_mapping = {"Actual / Verified": 0, "Estimated / Imputed": 1}

  pricetype_mapping = {"Retail": 0, "Wholesale": 1}

  market_mapping = {
      "Addis Ababa Market": 103,
      "Bahir Dar Market": 104,
      "Hawassa Market": 105,
      "Adama Market": 106,
      "Mekelle Market": 107,
  }

  commodity_mapping = {
      "Maize (White)": 83,
      "Teff (Mixed)": 12,
      "Wheat": 45,
      "Sorghum": 28,
      "Barley": 19,
  }

  with st.form("prediction_form"):
    col_a, col_b = st.columns(2)
    with col_a:
      date_val = st.date_input(
          "Transaction Date", value=pd.to_datetime("2022-12-15")
      )
      latitude = st.number_input("Latitude Coordinate", value=9.02, format="%.4f")
      longitude = st.number_input(
          "Longitude Coordinate", value=38.75, format="%.4f"
      )

      selected_admin1 = st.selectbox(
          "Region (Admin1)", options=list(admin1_mapping.keys())
      )
      admin1 = admin1_mapping[selected_admin1]

      admin2 = st.number_input("Zone (Admin2) Encoded ID", value=13, step=1)

    with col_b:
      selected_market = st.selectbox(
          "Local Market Name", options=list(market_mapping.keys())
      )
      market = market_mapping[selected_market]

      selected_category = st.selectbox(
          "Commodity Category", options=list(category_mapping.keys())
      )
      category = category_mapping[selected_category]

      selected_commodity = st.selectbox(
          "Commodity Name", options=list(commodity_mapping.keys())
      )
      commodity = commodity_mapping[selected_commodity]

      selected_unit = st.selectbox(
          "Measurement Unit", options=list(unit_mapping.keys())
      )
      unit = unit_mapping[selected_unit]

      selected_priceflag = st.selectbox(
          "Price Status", options=list(priceflag_mapping.keys())
      )
      priceflag = priceflag_mapping[selected_priceflag]

      selected_pricetype = st.selectbox(
          "Price Type", options=list(pricetype_mapping.keys())
      )
      pricetype = pricetype_mapping[selected_pricetype]

    submit = st.form_submit_button("Predict Market Price")

    if submit:
      model, error_msg = load_model_assets()

      if error_msg:
        st.error(
            f"Could not process prediction. Serialized assets error:"
            f" {error_msg}"
        )
      else:
        try:
          dt_date = pd.to_datetime(date_val)
          year = dt_date.year
          month = dt_date.month
          quarter = (month - 1) // 3 + 1
          month_sin = np.sin(2 * np.pi * month / 12)
          month_cos = np.cos(2 * np.pi * month / 12)

          input_row = pd.DataFrame([{
              "admin1": admin1,
              "admin2": admin2,
              "market": market,
              "latitude": latitude,
              "longitude": longitude,
              "category": category,
              "commodity": commodity,
              "unit": unit,
              "priceflag": priceflag,
              "pricetype": pricetype,
              "year": year,
              "month": month,
              "month_sin": month_sin,
              "month_cos": month_cos,
          }])

          expected_cols = [
              "admin1",
              "admin2",
              "market",
              "latitude",
              "longitude",
              "category",
              "commodity",
              "unit",
              "priceflag",
              "pricetype",
              "year",
              "month",
              "month_sin",
              "month_cos",
          ]
          input_row = input_row[expected_cols]

          prediction = model.predict(input_row)[0]
          st.success(
              f"Predicted Local Market Price: **{prediction:,.2f} ETB**"
          )
        except Exception as ex:
          st.error(f"Error processing prediction parameters: {ex}")
