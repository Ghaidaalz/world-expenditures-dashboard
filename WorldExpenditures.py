# WorldExpenditures.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Load Data
df = pd.read_csv("C:/week1 project/WorldExpenditures.csv")

# 1. Basic Dataset Overview
st.title("🌍 World Government Expenditure Dashboard")

# --- Dataset Overview ---
st.markdown("### 📊 World Government Expenditure Dataset")
with st.expander("ℹ️ About the Dataset"):
    st.markdown("""
    This dataset contains **world government expenditures** from **2000 to 2021** across various sectors.  
    It includes:

    - **Country** and **Year**  
    - **Sector** of spending (e.g., Health, Education, Military)  
    - **Expenditure** in million USD  
    - **Expenditure as a percentage of GDP**

    **What is GDP?**  
    GDP (Gross Domestic Product) is the total value of all goods and services produced by a country in a year.  
    When we say **"expenditure as a % of GDP,"** it shows how much a country spends on a sector relative to its total economy.

    This data enables analysis of:
    - Government spending patterns  
    - Sectoral priorities across nations  
    - Economic trends over time
    """)
# ✅ Preview a sample of the dataset
st.subheader("🔍 Sample of the Dataset")
st.dataframe(df.head())

# --- Dataset Overview (Before Cleaning) ---
st.header("📄 Dataset Overview (Full Data)")

# Show shape
st.markdown(f"**Shape:** {df.shape}")

# Show missing values
st.markdown("**Missing values:**")
st.dataframe(df.isna().sum().reset_index().rename(columns={'index': 'Column', 0: 'Missing Values'}))

# Show duplicate rows count
duplicate_count = df.duplicated().sum()
st.markdown(f"**Duplicate rows:** {duplicate_count}")

# --- Optional: Show Data Cleaning Steps ---
with st.expander("🧹 View Data Cleaning Steps"):
    st.markdown("""
    **Data Cleaning Summary:**
    - Removed rows with missing values using `dropna()`.
    - Dropped irrelevant or fully empty columns.
    - Excluded zero values when calculating mode to avoid skewed results.
    - Converted `Year` column to integer type for filtering and modeling.
    - Standardized column names and removed unnecessary whitespace.
    """)

# Clean Data
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df[df["Sector"].str.lower() != "total function"]
df = df.dropna(subset=["Year", "Expenditure(million USD)", "GDP(%)"])
df["Year"] = df["Year"].astype(int)
df["Expenditure(million USD)"] = df["Expenditure(million USD)"].astype(float)
df["GDP(%)"] = df["GDP(%)"].astype(float)


# Sidebar
st.sidebar.header("📁 Dataset Info")
st.sidebar.markdown("This dashboard analyzes world government expenditures.")
year_options = sorted(df["Year"].unique())
country_options = sorted(df["Country"].unique())
selected_year = st.sidebar.selectbox("Select a Year:", year_options)
selected_country = st.sidebar.selectbox("Select a Country:", country_options)

# Filtered Data
filtered_df = df[(df["Year"] == selected_year) & (df["Country"] == selected_country)]

# 2. Basic Statistics
st.subheader("📊Statistics (Selected Year & Country Only)")
if not filtered_df.empty:
    numeric_cols = filtered_df.select_dtypes(include=np.number).columns.drop("Year")
    st.write("Mean:")
    st.write(filtered_df[numeric_cols].mean())
    st.write("Median:")
    st.write(filtered_df[numeric_cols].median())
    st.write("Mode:")
    mode_result = filtered_df[numeric_cols].mode()
    st.write(mode_result.iloc[0] if not mode_result.empty else "No mode found")
else:
    st.warning("⚠️ No data available for the selected year and country.")

# 3. Visualizations
st.subheader("📈 Visualizations (for Selected Year & Country)")
if not filtered_df.empty:
    # Histogram (log-scaled)
    fig1, ax1 = plt.subplots()
    ax1.hist(np.log1p(filtered_df["Expenditure(million USD)"]), bins=10, color="skyblue")
    ax1.set_title("Log-Scaled Expenditure Distribution")
    ax1.set_xlabel("Log(Expenditure)")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)

    # Correlation Heatmap
    fig2, ax2 = plt.subplots()
    sns.heatmap(filtered_df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax2)
    ax2.set_title("Correlation Heatmap")
    st.pyplot(fig2)

# 4. Question 1: Top 5 Countries by Total Expenditure
st.header("❓ Top 5 Countries by Total Expenditure")
top5_countries = df.groupby("Country")["Expenditure(million USD)"].sum().sort_values(ascending=False).head(5)
st.dataframe(top5_countries)
fig3, ax3 = plt.subplots()
top5_countries.plot(kind="bar", ax=ax3)
ax3.set_title("Top 5 Countries by Total Expenditure")
ax3.set_ylabel("Expenditure (Million USD)")
st.pyplot(fig3)

# 5. Question 2: Top 5 Sectors by Total Expenditure
st.header("❓ Top 5 Sectors by Total Expenditure")
top5_sectors = df.groupby("Sector")["Expenditure(million USD)"].sum().sort_values(ascending=False).head(5)
st.dataframe(top5_sectors)
fig4, ax4 = plt.subplots()
top5_sectors.plot(kind="bar", ax=ax4)
ax4.set_title("Top 5 Sectors by Total Expenditure")
ax4.set_ylabel("Expenditure (Million USD)")
st.pyplot(fig4)

# 6. Question 3: Top 10 Highest % of GDP
st.header("❓ Which country spent the highest % of GDP on a single sector in any year?")
top_10 = df.sort_values("GDP(%)", ascending=False).head(10)
fig5, ax5 = plt.subplots(figsize=(10, 6))
bars = ax5.barh(
    top_10['Country'] + ' - ' + top_10['Sector'] + ' (' + top_10['Year'].astype(str) + ')',
    top_10["GDP(%)"], color="skyblue")
for bar in bars:
    ax5.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2, f'{bar.get_width():.2f}%', va='center')
ax5.set_title("Top 10 Expenditures as % of GDP")
ax5.set_xlabel("% of GDP")
ax5.set_ylabel("Country - Sector (Year)")
st.pyplot(fig5)
max_entry = df.loc[df["GDP(%)"].idxmax()]
st.markdown(f"**Most:** {max_entry['Country']} spent {max_entry['GDP(%)']}% of GDP on {max_entry['Sector']} in {max_entry['Year']}.")

# 7. Question 4: Education vs Health Over Years
st.header("❓ How does education expenditure compare to health over the years?")
pivot_df = df.pivot_table(index="Year", columns="Sector", values="Expenditure(million USD)", aggfunc="sum")
fig6, ax6 = plt.subplots(figsize=(10, 6))
pivot_df[["Education", "Health"]].plot(ax=ax6)
ax6.set_title("Education vs Health Expenditure Over the Years")
ax6.set_ylabel("Expenditure (Million USD)")
ax6.grid(True)
st.pyplot(fig6)

# 8. Question 5: Which sectors increased the most from 2000 to 2021
st.header("❓ Which sectors increased the most from 2000 to 2021?")
exp_2000 = df[df["Year"] == 2000].groupby("Sector")["Expenditure(million USD)"].sum()
exp_2021 = df[df["Year"] == 2021].groupby("Sector")["Expenditure(million USD)"].sum()
comparison = pd.DataFrame({"2000": exp_2000, "2021": exp_2021})
comparison["Increase"] = comparison["2021"] - comparison["2000"]
fig7, ax7 = plt.subplots()
comparison["Increase"].sort_values(ascending=False).plot(kind="bar", ax=ax7, color="green")
ax7.set_title("Increase in Expenditure by Sector (2000 to 2021)")
ax7.set_ylabel("Increase (Million USD)")
st.pyplot(fig7)

# 9. Question 6 (Old Q9): Most Stable Spending Sectors
st.header("📌 Most Stable Spending Sectors")
sector_variation = df.groupby("Sector")["Expenditure(million USD)"].std().sort_values()
fig_q6 = px.bar(sector_variation, title="Sectors with the Most Stable Spending (Lowest Std Dev)",
                labels={"value": "Standard Deviation", "index": "Sector"},
                color=sector_variation.values, color_continuous_scale="Blues")
st.plotly_chart(fig_q6)
st.markdown("**Insight:** Lower standard deviation means more consistent spending across years.")

# 10. Question 7 (Old Q10): Average GDP% per Sector Globally
st.header("📌 Average GDP% per Sector Globally")
avg_gdp_per_sector = df.groupby("Sector")["GDP(%)"].mean().sort_values(ascending=False)
fig_q7 = px.pie(names=avg_gdp_per_sector.index, values=avg_gdp_per_sector.values,
                title="Average %GDP Spent by Sector", hole=0.4,
                color_discrete_sequence=px.colors.sequential.RdBu)
st.plotly_chart(fig_q7)

# 11. Explore GDP% by Country
st.header("🌐 Explore GDP% by Sector and Country")
country_sector_gdp = df[df["Country"] == selected_country]
fig_country = px.bar(
    country_sector_gdp, x="Sector", y="GDP(%)", color="GDP(%)",
    title=f"GDP% by Sector - {selected_country}",
    labels={"GDP(%)": "% of GDP"}, color_continuous_scale="Viridis"
)
fig_country.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_country)

# ----------------------------
# Question 9: Top 5 Spending Sectors in a Specific Country (e.g., Germany) in 2020
# ----------------------------
st.markdown("### ❓ What were the top 5 spending sectors in Germany in 2021?")

example_country = 'Germany'
country_2020 = df[(df['Country'] == example_country) & (df['Year'] == 2021)]
top_sectors_country = country_2020.groupby('Sector')['GDP(%)'].sum().sort_values(ascending=False)

fig9, ax9 = plt.subplots(figsize=(10, 6))
top_sectors_country.head(5).plot(kind='bar', color='skyblue', ax=ax9)
ax9.set_title(f"Top 5 Spending Sectors in {example_country} (2021)")
ax9.set_ylabel("Spending (% of GDP)")
ax9.set_xticklabels(ax9.get_xticklabels(), rotation=45)
st.pyplot(fig9)


# ----------------------------
# Question 10: Top 3 Globally Funded Sectors in 2021
# ----------------------------
st.markdown("### ❓ Which 3 sectors received the highest global funding in 2021?")

global_2021 = df[df['Year'] == 2021]
top_global_sectors = global_2021.groupby('Sector')['GDP(%)'].sum().sort_values(ascending=False).head(3)

fig10, ax10 = plt.subplots(figsize=(6, 5))
top_global_sectors.plot(kind='bar', color='skyblue', ax=ax10)
ax10.set_title("Top 3 Globally Funded Sectors in 2021")
ax10.set_ylabel("Total GDP% Spending")
ax10.set_xticklabels(ax10.get_xticklabels(), rotation=45)
st.pyplot(fig10)

# ----------------------------
# Question 11: How has total government expenditure changed globally over time?
# ----------------------------
st.markdown("### ❓ How has total government expenditure changed globally over time?")

# Calculate total expenditure by year
yearly_totals = df.groupby('Year')['Expenditure(million USD)'].sum().reset_index()

# Create the plot
fig11, ax11 = plt.subplots(figsize=(10, 5))
ax11.plot(yearly_totals['Year'], yearly_totals['Expenditure(million USD)']/1e6,
          marker='o', linestyle='-', color='royalblue', linewidth=2)
ax11.set_title('Global Government Expenditure Trend (2000–2021)')
ax11.set_xlabel('Year')
ax11.set_ylabel('Total Expenditure (Trillion USD)')
ax11.grid(True, alpha=0.3)
st.pyplot(fig11)

# Display the summary table
st.markdown("#### 📊 Yearly Global Expenditure Summary:")
st.dataframe(yearly_totals.rename(columns={"Expenditure(million USD)": "Total Expenditure (Million USD)"}))



# --- Forecasting with Random Forest Regressor ---
st.header("📈 Random Forest Regression Forecast")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Clean and prepare data
df["Expenditure(million USD)"] = pd.to_numeric(df["Expenditure(million USD)"], errors='coerce')
df_clean = df.dropna(subset=["Expenditure(million USD)", "Year"])
df_health = df_clean[df_clean["Sector"] == "Health"]

# Group by year
yearly_health = df_health.groupby("Year")["Expenditure(million USD)"].sum().reset_index()

# Features and target
X = yearly_health[["Year"]]
y = yearly_health["Expenditure(million USD)"]

# Train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)
y_pred = rf_model.predict(X)

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(X["Year"], y, marker='o', label="Actual", linewidth=2)
plt.plot(X["Year"], y_pred, linestyle='--', marker='o', color='skyblue', label="Predicted", linewidth=2)
plt.title("Random Forest Regression Forecast")
plt.xlabel("Year")
plt.ylabel("Total Expenditure (Million USD)")
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Evaluate model
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

st.markdown(f"**Model R² Score:** {r2:.4f}")
st.markdown(f"**Mean Squared Error:** {mse:,.2f}")
