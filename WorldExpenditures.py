# WorldExpenditures.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="World Expenditure Dashboard", layout="wide")

# Load dataset
df = pd.read_csv("WorldExpenditures.csv")

# Format large numbers
def format_millions(x):
    if pd.isna(x):
        return "N/A"
    elif abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    elif abs(x) >= 1:
        return f"{x / 1_000:.2f}K"
    else:
        return f"{x:.2f}"


def render_styled_table(df):
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype in [np.float64, np.int64] and col != "GDP(%)" and col != "Year":
            df_copy[col] = df_copy[col].apply(format_millions)
    styled_html = df_copy.to_html(classes='styled-table', index=False)
    st.markdown(
        """
        <style>
        .styled-table {
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 14px;
            width: 100%;
            border: 1px solid #ccc;
            text-align: center;
        }
        .styled-table thead tr {
            background-color: #009879;
            color: #ffffff;
        }
        .styled-table th, .styled-table td {
            padding: 8px 12px;
            border: 1px solid #ddd;
        }
        .styled-table tbody tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        </style>
        """,
        unsafe_allow_html=True)
    st.markdown(styled_html, unsafe_allow_html=True)


# Title
st.title("🌍 World Government Expenditure Dashboard")
st.markdown("### 📊 World Government Expenditure Dataset")

with st.expander("ℹ️ About the Dataset"):
    st.markdown("""
    This dataset contains world government expenditures from 2000 to 2021 across various sectors.
    It includes:
    - Country and Year
    - Sector of spending (e.g., Health, Education, Military)
    - Expenditure in million USD
    - Expenditure as a percentage of GDP
    """)

# Data cleaning
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df[df["Sector"].str.lower() != "total function"]
df = df.dropna(subset=["Year", "Expenditure(million USD)", "GDP(%)"])
df["Year"] = df["Year"].astype(int)
df["Expenditure(million USD)"] = df["Expenditure(million USD)"].astype(float)
df["GDP(%)"] = df["GDP(%)"].astype(float)

# Sidebar filters
with st.sidebar:
    st.markdown(
        """
        <div style='border: 2px solid #f63366; padding: 10px; border-radius: 10px; background-color: #f9f9f9;'>
        <h4>📁 Dataset Info</h4>
        <p>This dashboard analyzes world government expenditures.</p>
        </div>
        """, unsafe_allow_html=True)
    selected_year = st.selectbox("Select a Year:", sorted(df["Year"].unique()))
    selected_country = st.selectbox("Select a Country:", sorted(df["Country"].unique()))

filtered_df = df[(df["Year"] == selected_year) & (df["Country"] == selected_country)]

# Dataset preview
st.subheader("🔍 Sample of the Dataset")
render_styled_table(df.head())

# Dataset overview
st.header("📄 Dataset Overview (Full Data)")
st.markdown(f"**Shape:** {df.shape}")

st.markdown("**Missing values:**")
render_styled_table(df.isna().sum().reset_index().rename(columns={'index': 'Column', 0: 'Missing Values'}))

duplicate_count = df.duplicated().sum()
st.markdown(f"**Duplicate rows:** {duplicate_count}")

with st.expander("🧹 View Data Cleaning Steps"):
    st.markdown("""
    - Dropped missing and unnecessary columns  
    - Removed rows with missing values  
    - Converted data types to numeric
    """)

#Statistics Section
st.subheader("📊 Statistics (Selected Year & Country)")

if not filtered_df.empty:
    numeric_cols = filtered_df.select_dtypes(include=np.number).columns.drop("Year")

    #Mean
    mean_df = filtered_df[numeric_cols].mean().round(2).rename("Value").reset_index()
    mean_df.columns = ["Metric", "Value"]
    mean_df["Value"] = mean_df["Value"].apply(format_millions)

    #Median
    median_df = filtered_df[numeric_cols].median().round(2).rename("Value").reset_index()
    median_df.columns = ["Metric", "Value"]
    median_df["Value"] = median_df["Value"].apply(format_millions)

    #Mode
    mode_expenditure = filtered_df.loc[filtered_df["Expenditure(million USD)"] > 0, "Expenditure(million USD)"].mode()
    mode_gdp = filtered_df.loc[filtered_df["GDP(%)"] > 0, "GDP(%)"].mode()

    mode_data = {
        "Metric": ["Expenditure(million USD)", "GDP(%)"],
        "Value": [
            format_millions(mode_expenditure[0]) if not mode_expenditure.empty else "N/A",
            format_millions(mode_gdp[0]) if not mode_gdp.empty else "N/A"
        ]
    }
    mode_df = pd.DataFrame(mode_data)

    #Display all 3 tables side by side
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<h5 style='color:#F9C74F;'>Mean</h5>", unsafe_allow_html=True)
        st.markdown(mean_df.to_html(classes="custom-yellow-table", index=False), unsafe_allow_html=True)

    with col2:
        st.markdown("<h5 style='color:#F9C74F;'>Median</h5>", unsafe_allow_html=True)
        st.markdown(median_df.to_html(classes="custom-yellow-table", index=False), unsafe_allow_html=True)

    with col3:
        st.markdown("<h5 style='color:#F9C74F;'>Mode</h5>", unsafe_allow_html=True)
        st.markdown(mode_df.to_html(classes="custom-yellow-table", index=False), unsafe_allow_html=True)

else:
    st.warning("⚠️ No data available for this selection.")

# 📊 Sector-wise Expenditure Distribution – Funnel Chart (Horizontal)
st.subheader("📊 Sector-wise Expenditure Distribution")

if not filtered_df.empty:
    funnel_data = filtered_df.groupby("Sector")["Expenditure(million USD)"].sum().reset_index()
    funnel_data = funnel_data[funnel_data["Expenditure(million USD)"] > 0]
    funnel_data = funnel_data.sort_values(by="Expenditure(million USD)", ascending=True)

    # Apply formatting for display
    funnel_data["Formatted Value"] = funnel_data["Expenditure(million USD)"].apply(format_millions)

    if not funnel_data.empty:
        fig_funnel = px.funnel(
            funnel_data,
            x="Expenditure(million USD)",
            y="Sector",
            title=f"{selected_country} – Expenditure by Sector ({selected_year})",
            template="plotly_white",
            color_discrete_sequence=["#fdd835"] * len(funnel_data),  # yellow color
            text="Formatted Value"  # Add formatted labels
        )
        fig_funnel.update_traces(textposition="inside", textfont_size=14)
        fig_funnel.update_layout(
            width=900,
            height=700,
            margin=dict(t=70, l=100, r=50, b=50),
            font=dict(size=14)
        )
        st.plotly_chart(fig_funnel)
    else:
        st.info("No valid expenditure values to display.")
else:
    st.info("No data available for the selected filters.")


# Q1: Top 5 countries (Plotly)
st.header("❓Top 5 Countries by Total Expenditure")
top5_countries = df.groupby("Country")["Expenditure(million USD)"].sum().nlargest(5).reset_index()
render_styled_table(top5_countries)
fig_q1 = px.bar(top5_countries, x="Country", y="Expenditure(million USD)", color="Expenditure(million USD)",
                color_continuous_scale="blues", title="Top 5 Countries by Expenditure")
st.plotly_chart(fig_q1)

# Q2: Top 5 sectors (Plotly)
st.header("❓Top 5 Sectors by Total Expenditure")
top5_sectors = df.groupby("Sector")["Expenditure(million USD)"].sum().nlargest(5).reset_index()
render_styled_table(top5_sectors)
fig_q2 = px.bar(top5_sectors, x="Sector", y="Expenditure(million USD)", color="Expenditure(million USD)",
                color_continuous_scale="greens", title="Top 5 Sectors by Expenditure")
fig_q2.update_layout(xaxis_tickangle=-30)
st.plotly_chart(fig_q2)

# Q3: Highest % of GDP
st.header("❓Highest % of GDP on a Sector")
top_10 = df.sort_values("GDP(%)", ascending=False).head(10)
top_10["GDP(%)"] = top_10["GDP(%)"].round(2)  # <-- rounding GDP(%)
render_styled_table(top_10[["Country", "Year", "Sector", "GDP(%)"]])
max_entry = df.loc[df["GDP(%)"].idxmax()]
st.success(f"**Most:** {max_entry['Country']} spent {max_entry['GDP(%)']:.5f}% of GDP on {max_entry['Sector']} in {max_entry['Year']}.")

# Q4: Education vs Health
st.header("❓Education vs Health Spending Over Years")
pivot_df = df.pivot_table(index="Year", columns="Sector", values="Expenditure(million USD)", aggfunc="sum")
fig4 = px.line(pivot_df[["Education", "Health"]].reset_index(), x="Year", y=["Education", "Health"],
               markers=True, title="Education vs Health Expenditure Over the Years")
st.plotly_chart(fig4)

# Q5: Sector growth
st.header("❓Sector Growth from 2000 to 2021")
exp_2000 = df[df["Year"] == 2000].groupby("Sector")["Expenditure(million USD)"].sum()
exp_2021 = df[df["Year"] == 2021].groupby("Sector")["Expenditure(million USD)"].sum()
comparison = pd.DataFrame({"2000": exp_2000, "2021": exp_2021})
comparison["Increase"] = comparison["2021"] - comparison["2000"]
comparison = comparison.sort_values("Increase", ascending=False).reset_index()
render_styled_table(comparison)
fig_q5 = px.bar(comparison, x="Sector", y="Increase", color="Increase",
                color_continuous_scale="Oranges", title="Sector Growth from 2000 to 2021")
fig_q5.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_q5)

# Q6: Most stable sectors (keep as is)
st.header("📌 Most Stable Spending Sectors")
sector_variation = df.groupby("Sector")["Expenditure(million USD)"].std().sort_values()
fig_q6 = px.bar(sector_variation, title="Lowest Std Deviation in Sector Spending",
                labels={"value": "Standard Deviation", "index": "Sector"},
                color=sector_variation.values, color_continuous_scale="Purples")
st.plotly_chart(fig_q6)

# Q7: Avg GDP% per Sector (keep as is)
st.header("📌 Average GDP% per Sector (Globally)")
avg_gdp_per_sector = df.groupby("Sector")["GDP(%)"].mean().sort_values(ascending=False)
fig_q7 = px.pie(names=avg_gdp_per_sector.index, values=avg_gdp_per_sector.values,
                title="Average %GDP Spent by Sector", hole=0.4,
                color_discrete_sequence=px.colors.sequential.RdBu)
st.plotly_chart(fig_q7)

# Q8: GDP% by Sector for selected country (keep as is)
st.header(f"🌐 GDP% by Sector for {selected_country}")
country_sector_gdp = df[df["Country"] == selected_country]
fig_country = px.bar(country_sector_gdp, x="Sector", y="GDP(%)", color="GDP(%)",
                     title=f"{selected_country} - GDP% by Sector", color_continuous_scale="Viridis")
fig_country.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_country)

# Q9: Germany 2021 (interactive)
st.header("🇩🇪 Top 5 Spending Sectors in Germany (2021)")
germany_2021 = df[(df['Country'] == 'Germany') & (df['Year'] == 2021)]
top_sectors_ger = germany_2021.groupby('Sector')["GDP(%)"].sum().nlargest(5).reset_index()
fig9 = px.bar(top_sectors_ger, x="Sector", y="GDP(%)", color="GDP(%)",
              title="Top 5 Spending Sectors in Germany (2021)", color_continuous_scale="YlOrBr")
st.plotly_chart(fig9)

# Q10: Global Top 3 sectors in 2021 (interactive)
st.header("🌍 Top 3 Globally Funded Sectors in 2021")
global_2021 = df[df['Year'] == 2021]
top_sectors_2021 = global_2021.groupby('Sector')["GDP(%)"].sum().nlargest(3).reset_index()
fig10 = px.bar(top_sectors_2021, x="Sector", y="GDP(%)", color="GDP(%)",
               title="Top 3 Funded Sectors Globally in 2021", color_continuous_scale="Plasma")
st.plotly_chart(fig10)

# Q11: Global Expenditure Trend
st.header("📈 Global Expenditure Trend (2000–2021)")
yearly_total = df.groupby("Year")["Expenditure(million USD)"].sum().reset_index()
render_styled_table(yearly_total)
fig11 = px.line(yearly_total, x="Year", y="Expenditure(million USD)", markers=True,
                title="Total Global Expenditure Trend", line_shape='spline')
st.plotly_chart(fig11)

# Forecasting
st.header("🔮 Forecasting Health Expenditure (Random Forest)")
df_health = df[df["Sector"] == "Health"]
health_agg = df_health.groupby("Year")["Expenditure(million USD)"].sum().reset_index()
X = health_agg[["Year"]]
y = health_agg["Expenditure(million USD)"]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
y_pred = model.predict(X)

fig_rf = px.line(health_agg, x="Year", y="Expenditure(million USD)", markers=True, labels={"value": "Expenditure"},
                 title="Forecast vs Actual: Health Expenditure")
fig_rf.add_scatter(x=X["Year"], y=y_pred, mode="lines+markers", name="Predicted")
st.plotly_chart(fig_rf)

r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
st.markdown(f"**R² Score:** {r2:.4f}")
st.markdown(f"**Mean Squared Error:** {format_millions(mse)}")
