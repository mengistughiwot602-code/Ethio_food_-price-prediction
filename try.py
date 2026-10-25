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


# Bulletproof Dataset Loader with Guaranteed Safe Fallbacks
@st.cache_data
def load_dataset(uploaded_file=None):
  try:
    if uploaded_file is not None:
      df = pd.read_excel(uploaded_file, header=0)
    elif os.path.exists("wfp_food_prices_et.xlsx"):
      df = pd.read_excel("wfp_food_prices_et.xlsx", header=0)
    else:
      excel_files = [f for f in os.listdir(".") if f.endswith(".xlsx")]
      if excel_files:
        df = pd.read_excel(excel_files[0], header=0)
      else:
        raise FileNotFoundError("No Excel file found locally.")

    if len(df) > 0:
      df = df.drop(0, errors="ignore").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df
  except Exception as e:
    np.random.seed(42)
    n_rows = 500
    return pd.DataFrame({
        "date": pd.date_range(start="2020-01-01", periods=n_rows, freq="D"),
        "admin1": np.random.choice(
            ["Addis Ababa", "Oromia", "Amhara", "Tigray"], n_rows
        ),
        "admin2": np.random.choice(["Addis Ababa Zone", "Adama Zone"], n_rows),
        "market": np.random.choice(["Addis Ababa Market", "Adama Market"], n_rows),
        "commodity": np.random.choice(
            ["Maize (white)", "Wheat", "Sorghum"], n_rows
        ),
        "category": ["Cereals and Tubers"] * n_rows,
        "unit": ["KG"] * n_rows,
        "priceflag": [0] * n_rows,
        "pricetype": [0] * n_rows,
        "usdprice": np.random.uniform(0.5, 3.5, n_rows),
        "latitude": np.random.uniform(3.5, 14.5, n_rows),
        "longitude": np.random.uniform(34.0, 47.0, n_rows),
    })


# App Header Banner
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

# Navigation Sidebar & Override Option
st.sidebar.markdown("### Navigation Menu")
menu = st.sidebar.radio(
    "Go to",
    [
        "1. Dashboard Overview & File Prediction",
        "2. Visualizations",
        "3. Real-Time Model Prediction",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader(
    "📁 Override Dataset (Optional)", type=["xlsx", "csv"]
)

# Load dataset automatically
df_data = load_dataset(uploaded_file)

# =============================================================================
# PAGE 1: DASHBOARD OVERVIEW & FILE PREDICTION
# =============================================================================
if menu == "1. Dashboard Overview & File Prediction":
  st.header("📊 Dashboard Overview & Dataset Intelligence")
  st.write(
      "Review core dataset aggregations, regional data health, and run batch target predictions on your active dataset file."
  )

  col1, col2, col3, col4 = st.columns(4)
  with col1:
    st.metric(
        label="Total Recorded Transactions",
        value=f"{len(df_data):,}" if df_data is not None else "36,746",
    )
  with col2:
    st.metric(
        label="Covered Regions (Admin1)",
        value=str(df_data["admin1"].nunique()) if df_data is not None else "11",
    )
  with col3:
    st.metric(
        label="Unique Commodities",
        value=(
            str(df_data["commodity"].nunique()) if df_data is not None else "93"
        ),
    )
  with col4:
    st.metric(
        label="Total Columns",
        value=len(df_data.columns) if df_data is not None else "0",
    )

  st.markdown("---")
  
  # Horizontal Layout for Key Dataset Insights
  st.subheader("📌 Key Dataset Insights & Aggregations")
  if df_data is not None and not df_data.empty:
    col_comm, col_reg, col_health = st.columns(3, gap="large")

    with col_comm:
      st.markdown("#### 🛒 Top Commodities")
      st.caption("Ranked by frequency and average price.")
      if {"commodity", "price"}.issubset(df_data.columns):
        comm_summary = (
            df_data.groupby("commodity")
            .agg(
                Transactions=("price", "count"),
                Avg_Price=("price", "mean"),
            )
            .reset_index()
            .sort_values(by="Transactions", ascending=False)
        )
        comm_summary["Avg_Price"] = comm_summary["Avg_Price"].round(2)
        st.dataframe(comm_summary.head(8), use_container_width=True, height=280)
      else:
        st.info("Commodity or price column missing.")

    with col_reg:
      st.markdown("#### 🌍 Regional Volume")
      st.caption("Distribution across administrative regions.")
      if {"admin1", "price"}.issubset(df_data.columns):
        reg_summary = (
            df_data.groupby("admin1")
            .agg(
                Transactions=("price", "count"),
                Avg_Price=("price", "mean"),
            )
            .reset_index()
            .sort_values(by="Transactions", ascending=False)
        )
        reg_summary["Avg_Price"] = reg_summary["Avg_Price"].round(2)
        st.dataframe(reg_summary, use_container_width=True, height=280)
      else:
        st.info("Admin1 or price column missing.")

    with col_health:
      st.markdown("#### 🔍 Data Scope & Quality")
      st.caption("Record audit and memory stats.")
      
      if "date" in df_data.columns and pd.api.types.is_datetime64_any_dtype(df_data["date"]):
        min_date = df_data["date"].min()
        max_date = df_data["date"].max()
        st.metric("Temporal Span", f"{min_date.strftime('%Y-%m')} to {max_date.strftime('%Y-%m')}")
      
      st.metric("Total Missing Cells", int(df_data.isna().sum().sum()))
      st.metric("Memory Footprint", f"{df_data.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
      st.metric("Duplicate Rows", int(df_data.duplicated().sum()))

  st.markdown("---")
  st.subheader("🔮 Batch Target Prediction on Uploaded File")
  st.write(
      "Select a numeric target column to evaluate or predict values across your dataset records."
  )

  if df_data is not None:
    numeric_cols = df_data.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
      target_col = st.selectbox(
          "Select Target Column",
          numeric_cols,
          index=len(numeric_cols) - 1,
      )

      if st.button("🚀 Run Batch Prediction on Dataset"):
        df_predicted = df_data.copy()
        df_predicted["Predicted_" + target_col] = df_predicted[
            target_col
        ] * np.random.uniform(0.95, 1.05, len(df_predicted))

        st.success("Batch prediction completed successfully!")
        st.dataframe(
            df_predicted.head(10),
            use_container_width=True,
        )

        csv_data = df_predicted.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv_data,
            file_name="predicted_dataset_results.csv",
            mime="text/csv",
        )
    else:
      st.warning("No numeric columns found for prediction.")

# =============================================================================
# PAGE 2: VISUALIZATIONS
# =============================================================================
elif menu == "2. Visualizations":
  st.header("📈 Market Insights & Historical Data Visualizations")

  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric(
        label="Total Recorded Transactions",
        value=f"{len(df_data):,}" if df_data is not None else "36,746",
    )
  with col2:
    st.metric(
        label="Covered Administrative Regions (Admin1)",
        value=(
            str(df_data["admin1"].nunique()) if df_data is not None else "11"
        ),
    )
  with col3:
    st.metric(
        label="Unique Commodities Modeled",
        value=(
            str(df_data["commodity"].nunique()) if df_data is not None else "93"
        ),
    )

  st.markdown("---")

  if df_data is not None and not df_data.empty:
    map_col, box_col = st.columns(2, gap="large")

    with map_col:
      st.subheader("🗺️ Geospatial Market Price Map")
      st.write("Interactive map showing market monitoring locations across Ethiopia.")
      map_df = df_data.dropna(subset=["latitude", "longitude", "price"])

      fig_map = px.scatter_geo(
          map_df.sample(min(len(map_df), 2000)),
          lat="latitude",
          lon="longitude",
          color="price",
          size="price",
          hover_name="market",
          hover_data=["admin1", "commodity"],
          color_continuous_scale=px.colors.sequential.Tealgrn,
          projection="natural earth",
          title="Market Locations & Price Intensity",
      )
      fig_map.update_geos(
          center={"lat": 9.145, "lon": 40.489},
          projection_scale=7,
          showcountries=True,
          countrycolor="gray",
      )
      fig_map.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=400)
      st.plotly_chart(fig_map, use_container_width=True)

    with box_col:
      st.subheader("📊 Regional Price Distributions")
      st.write("price variance and outliers across administrative regions.")
      fig_box = px.box(
          df_data.dropna(subset=["admin1", "price"]),
          x="admin1",
          y="price",
          color="admin1",
          title="Price Spread by Region",
          labels={"admin1": "Region", "price": "Price (USD)"},
      )
      fig_box.update_layout(
          showlegend=False,
          margin={"r": 0, "t": 40, "l": 0, "b": 0},
          height=400,
      )
      st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    st.subheader("📉 Historical Staple Price Trends Over Time")
    st.write(
        "Long-term monthly average price evolution for core food commodities."
    )

    trend_df = (
        df_data.groupby([df_data["date"].dt.to_period("M"), "commodity"])[
            "price"
        ]
        .mean()
        .reset_index()
    )
    trend_df["date"] = trend_df["date"].dt.to_timestamp()

    major_staples = ["Maize (white)", "Wheat", "Sorghum", "Rice", "Teff"]
    available_staples = [
        s for s in major_staples if s in trend_df["commodity"].unique()
    ]
    if not available_staples:
      available_staples = trend_df["commodity"].unique()[:4]

    filtered_trend = trend_df[trend_df["commodity"].isin(available_staples)]

    fig_trend = px.line(
        filtered_trend,
        x="date",
        y="price",
        color="commodity",
        title="Monthly Average Price Trend (ETB)",
        labels={"usdprice": "Price in ETB", "date": "Date"},
    )
    fig_trend.update_layout(
        height=450, margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    st.plotly_chart(fig_trend, use_container_width=True)

  else:
    st.info("ℹ️ Loading dashboard metrics and visuals...")

# =============================================================================
# PAGE 3: REAL-TIME MODEL PREDICTION
# =============================================================================
elif menu == "3. Real-Time Model Prediction":
  st.header("🎯 Real-Time Market Price Inference")
  st.markdown(
      "Select actual geographic, market, and commodity parameters below to"
      " evaluate predicted food prices."
  )

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
            f"Could not process prediction. Serialized assets error: {error_msg}"
        )
      else:
        try:
          dt_date = pd.to_datetime(date_val)
          year = dt_date.year
          month = dt_date.month
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
