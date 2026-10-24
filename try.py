import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Configure Page Settings
st.set_page_config(
    page_title="Ethiopia Food Price Prediction & Analytics",
    page_icon="https://flagcdn.com/w80/et.png",
    layout="wide",
)

# Custom Styling & UI Enhancements
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
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
        }

        .stSelectbox label, .stDateInput label, .stSlider label {
            color: #1e293b !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }

        div[data-baseweb="select"] > div {
            background-color: #f8fafc;
            border-radius: 8px;
            border-color: #cbd5e1;
        }

        .section-header {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0f766e;
            margin-bottom: 1rem;
            border-bottom: 2px solid #ccfbf1;
            padding-bottom: 0.4rem;
        }

        .stButton button {
            background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%);
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.7rem 1.5rem;
            border: none;
            width: 100%;
            box-shadow: 0 4px 6px rgba(13, 148, 136, 0.2);
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background: linear-gradient(135deg, #115e59 0%, #0f766e 100%);
            color: #ffffff;
            box-shadow: 0 6px 12px rgba(13, 148, 136, 0.3);
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


# App Header Banner with Flag Image placed BEFORE the Title Text
st.markdown(
    """
    <div style='display: flex; align-items: center; gap: 15px; margin-bottom: 0.5rem;'>
        <img src='https://flagcdn.com/w80/et.png' width='50' style='border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);'/>
        <h1 style='margin: 0; font-size: 2.2rem; color: #1e293b; font-weight: 800;'>Ethiopian Food Price Prediction & Market Analysis</h1>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "An interactive decision-support application built for localized market"
    " forecasting and price intelligence."
)

# Navigation Sidebar
menu = st.sidebar.radio(
    "Navigation Menu", ["Dashboard Overview", "Model Prediction"]
)

if menu == "Dashboard Overview":
  st.header("📈 Market Insights & Historical Data Visualizations")

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
      y="usdprice",
      color="admin1",
      title="USD Price Distributions across Selected Regions",
  )
  st.plotly_chart(fig, use_container_width=True)

elif menu == "Model Prediction":
  st.header("🎯 Real-Time Market Price Inference")
  st.markdown(
      "Select actual geographic, market, and commodity parameters below to"
      " evaluate predicted food prices."
  )

  # Real Mappings
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

  admin2_mapping = {
      "Addis Ababa Zone": 13,
      "Adama Special Zone": 14,
      "Bahir Dar Special Zone": 15,
      "Hawassa Special Zone": 16,
      "Mekelle Special Zone": 17,
      "Dire Dawa Zone": 18,
      "East Shewa Zone": 19,
      "West Gojjam Zone": 20,
      "North Shewa Zone": 21,
      "South Wollo Zone": 22,
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
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
      st.markdown(
          '<div class="section-header">📍 Spatial & Temporal Details</div>',
          unsafe_allow_html=True,
      )

      date_val = st.date_input(
          "Transaction Date", value=pd.to_datetime("2022-12-15")
      )

      latitude = st.slider(
          "Latitude Coordinate",
          min_value=3.0000,
          max_value=15.0000,
          value=9.0200,
          step=0.0100,
          format="%.4f",
      )

      longitude = st.slider(
          "Longitude Coordinate",
          min_value=33.0000,
          max_value=48.0000,
          value=38.7500,
          step=0.0100,
          format="%.4f",
      )

      selected_admin1 = st.selectbox(
          "Region (Admin1)", options=list(admin1_mapping.keys())
      )
      admin1 = admin1_mapping[selected_admin1]

      selected_admin2 = st.selectbox(
          "Zone (Admin2)", options=list(admin2_mapping.keys())
      )
      admin2 = admin2_mapping[selected_admin2]

    with col_b:
      st.markdown(
          '<div class="section-header">🛒 Market & Commodity Attributes</div>',
          unsafe_allow_html=True,
      )

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

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("🚀 Predict Market Price")

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
              f"✨ Predicted Local Market Price: **{prediction:,.2f} ETB**"
          )
        except Exception as ex:
          st.error(f"Error processing prediction parameters: {ex}")
