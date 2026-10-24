import os

# Writing app.py with robust native emojis instead of external image dependencies to prevent blank screen
app_code = """import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import joblib

# Configure Page Settings
st.set_page_config(
    page_title="Ethiopia Food Price Predictor & Dashboard",
    page_icon="🇪🇹",
    layout="wide"
)

# Stylized Header Banner
st.markdown(
    \"\"\"
    <div style='display: flex; align-items: center; gap: 15px; margin-bottom: 1rem;'>
        <div style='font-size: 2.5rem;'>🇪🇹</div>
        <h1 style='margin: 0; font-size: 2.2rem; color: #1e293b; font-weight: 800;'>Ethiopian Food Price Analytics & Forecasting</h1>
    </div>
    \"\"\",
    unsafe_allow_html=True
)
st.markdown("An interactive decision-support application built for localized market forecasting, spatial price mapping, and historical monitoring.")

# Custom CSS Styling to elevate Streamlit interface components
st.markdown(
    \"\"\"
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        [data-testid=\"stSidebar\"] {
            background-color: #f4f6f9;
        }
        [data-testid=\"stForm\"] {
            background-color: #ffffff;
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
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
    \"\"\",
    unsafe_allow_html=True
)

# Cache-loaded database setup
@st.cache_data
def load_data():
    if os.path.exists("wfp_food_prices_et.xlsx"):
        df = pd.read_excel("wfp_food_prices_et.xlsx")
        df['date'] = pd.to_datetime(df['date'])
        return df
    else: 
        # Fallback synthetic generator to keep dashboard operational in testing environments
        dates = pd.date_range(start="2000-01-15", end="2022-12-15", freq="M")
        regions = ["Addis Ababa", "Oromia", "Amhara", "Tigray", "Somali"]
        commodities = ["Maize (white)", "Wheat", "Sorghum", "Rice", "Teff"]
        data = []
        for dt in dates:
            for r in regions:
                for c in commodities:
                    data.append({
                        "date": dt,
                        "admin1": r,
                        "market": f"{r} Market",
                        "latitude": 9.0200 + np.random.uniform(-1, 1),
                        "longitude": 38.7500 + np.random.uniform(-1, 1),
                        "usdprice": np.random.uniform(10, 150),
                        "commodity": c
                    })
        return pd.DataFrame(data)

df_data = load_data()

# Navigation System
menu = st.sidebar.radio("Navigation Menu", ["Dashboard Overview", "Real-Time Model Inference"])

if menu == "Dashboard Overview":
    st.header("📈 Historical Market Price Analytics & Geospatial Dashboard")

    # Dynamic Metrics Row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Recorded Transactions", value=f"{len(df_data):,}")
    with col2:
        st.metric(label="Covered Administrative Regions (Admin1)", value=str(df_data['admin1'].nunique()))
    with col3:
        st.metric(label="Unique Food Commodities Modeled", value=str(df_data['commodity'].nunique()))

    st.markdown("---")

    # Layout Plot Row 1: Geospatial scatter map & Regional Price boxplot side-by-side
    map_col, box_col = st.columns(2, gap="large")

    with map_col:
        st.subheader("🗺️ Geospatial Market Price Map")
        st.write("Dynamic location intensity map tracking monitoring prices across Ethiopia.")
        map_df = df_data.dropna(subset=['latitude', 'longitude', 'usdprice'])

        # Performance Optimization: sample maximum of 3000 rows for rendering
        sample_map_df = map_df.sample(min(len(map_df), 3000), random_state=42)

        fig_map = px.scatter_mapbox( 
            sample_map_df,
            lat='latitude',
            lon='longitude',
            color='usdprice',
            size='usdprice',
            hover_name='market',
            hover_data=['admin1', 'commodity', 'usdprice'],
            color_continuous_scale=px.colors.sequential.Tealgrn,
            zoom=4.5,
            mapbox_style='carto-positron',
            title='Geographical Price Intensities across Ethiopia'
        )
        fig_map.update_layout(margin={'r': 0, 't': 40, 'l': 0, 'b': 0}, height=450)
        st.plotly_chart(fig_map, use_container_width=True)

    with box_col:
        st.subheader("📊 Regional Price Distributions")
        st.write("Spread and median price differences across administrative units (USD).")
        fig_box = px.box(
            df_data.dropna(subset=['admin1', 'usdprice']),
            x='admin1',
            y='usdprice',
            color='admin1',
            title='USD Price Spread by Region',
            labels={'admin1': 'Region', 'usdprice': 'Price (USD)'}
        )
        fig_box.update_layout(
            showlegend=False,
            margin={'r': 0, 't': 40, 'l': 0, 'b': 0},
            height=450
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    # Layout Plot Row 2: Staple Commodity Price Trends Over Time
    st.subheader("📉 Historical Staple Price Trends over Time")
    st.write("Historical monthly average price movements (USD) of core staples.")

    trend_df = df_data.groupby([df_data['date'].dt.to_period('M'), 'commodity'])['usdprice'].mean().reset_index()
    trend_df['date'] = trend_df['date'].dt.to_timestamp()

    major_staples = ['Maize (white)', 'Wheat', 'Sorghum', 'Rice', 'Teff']
    available_staples = [s for s in major_staples if s in trend_df['commodity'].unique()]
    if not available_staples:
        available_staples = list(trend_df['commodity'].unique()[:5])

    filtered_trend = trend_df[trend_df['commodity'].isin(available_staples)]

    fig_trend = px.line(
        filtered_trend,
        x='date',
        y='usdprice',
        color='commodity',
        title='Monthly Average Price Progression for Key Staples',
        labels={'usdprice': 'Average Price (USD)', 'date': 'Timeline'}
    )
    fig_trend.update_layout(height=450, margin={'r': 0, 't': 40, 'l': 0, 'b': 0})
    st.plotly_chart(fig_trend, use_container_width=True)

elif menu == "Real-Time Model Inference":
    st.header("🎯 Market Price Predictor Inference")
    st.markdown("Adjust the interactive input vectors below to calculate food market predictions utilizing our Random Forest core.")

    admin1_mapping = {'Addis Ababa': 7, 'Oromia': 8, 'Amhara': 2, 'Tigray': 10, 'Somali': 5, 'SNNPR / Sidama': 6, 'Afar': 1, 'Dire Dawa': 11}
    admin2_mapping = {'Addis Ababa Zone': 13, 'Adama Special Zone': 14, 'Bahir Dar Special Zone': 15, 'Hawassa Special Zone': 16, 'Mekelle Special Zone': 17, 'Dire Dawa Zone': 18, 'East Shewa Zone': 19, 'West Gojjam Zone': 20, 'North Shewa Zone': 21, 'South Wollo Zone': 22}
    category_mapping = {'Cereals and Tubers': 0, 'Pulses and Nuts': 1, 'Vegetables and Fruits': 2, 'Meat, Fish and Eggs': 3}
    unit_mapping = {'KG': 0, 'Quintal': 1, 'Pound': 2, 'Litre': 3}
    priceflag_mapping = {'Actual / Verified': 0, 'Estimated / Imputed': 1}
    pricetype_mapping = {'Retail': 0, 'Wholesale': 1}
    market_mapping = {'Addis Ababa Market': 103, 'Bahir Dar Market': 104, 'Hawassa Market': 105, 'Adama Market': 106, 'Mekelle Market': 107}
    commodity_mapping = {'Maize (White)': 83, 'Teff (Mixed)': 12, 'Wheat': 45, 'Sorghum': 28, 'Barley': 19}

    with st.form("prediction_form"):
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            st.subheader("🗺️ Spatial & Temporal Context")
            date_val = st.date_input("Transaction Date", value=pd.to_datetime("2022-12-15"))
            latitude = st.slider("Latitude Coordinates", 3.0, 15.0, 9.02, 0.01)
            longitude = st.slider("Longitude Coordinates", 33.0, 48.0, 38.75, 0.01)

            selected_admin1 = st.selectbox("Administrative Region (Admin1)", list(admin1_mapping.keys()))
            admin1 = admin1_mapping[selected_admin1]

            selected_admin2 = st.selectbox("Zone (Admin2)", list(admin2_mapping.keys()))
            admin2 = admin2_mapping[selected_admin2]

        with col_b:
            st.subheader("📊 Product & Channel Classification")
            selected_market = st.selectbox("Market Location", list(market_mapping.keys()))
            market = market_mapping[selected_market]

            selected_category = st.selectbox("Commodity Category Group", list(category_mapping.keys()))
            category = category_mapping[selected_category]

            selected_commodity = st.selectbox("Commodity Item", list(commodity_mapping.keys()))
            commodity = commodity_mapping[selected_commodity]

            selected_unit = st.selectbox("Product Unit", list(unit_mapping.keys()))
            unit = unit_mapping[selected_unit]

            selected_priceflag = st.selectbox("Price Assessment Type", list(priceflag_mapping.keys()))
            priceflag = priceflag_mapping[selected_priceflag]

            selected_pricetype = st.selectbox("Market Level Type", list(pricetype_mapping.keys()))
            pricetype = pricetype_mapping[selected_pricetype]

        submit = st.form_submit_button(" Estimate Price (ETB)")

        if submit:
            try:
                model = joblib.load("random_forest_model.joblib")

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
                    "admin1", "admin2", "market", "latitude", "longitude",
                    "category", "commodity", "unit", "priceflag", "pricetype",
                    "year", "month", "month_sin", "month_cos"
                ]
                input_row = input_row[expected_cols]

                prediction = model.predict(input_row)[0]
                st.success(f" Estimated Food Price: **{prediction:,.2f} ETB**")
            except Exception as e:
                st.error(f"Inference Processing Error: Ensure 'random_forest_model.joblib' is successfully serialized. (Details: {e})")
"""

with open('app.py', 'w') as f:
    f.write(app_code)

print("Successfully saved robust, offline-safe Streamlit dashboard code to app.py!")
