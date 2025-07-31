import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title and description
st.title("InnoStock: Big Data-Powered MAUT Stock Selection System")
st.markdown("""
This app evaluates and ranks stocks using **Multi-Attribute Utility Theory (MAUT)**.
Normalization is based on min-max scaling as per MAUT standard steps.
""")

# Upload stock dataset
uploaded_file = st.file_uploader("Upload Excel or CSV file with stock data", type=["csv", "xlsx"])

# Fallback sample data
def load_example():
    data = {
        'Stock': ['A', 'B', 'C'],
        'Price': [100, 120, 95],
        'P/E Ratio': [15, 18, 12],
        'Dividend Yield': [2.5, 3.0, 2.8],
        'Growth Rate': [8, 7, 9]
    }
    return pd.DataFrame(data)

# Load uploaded or sample data
df = pd.read_csv(uploaded_file) if uploaded_file and uploaded_file.name.endswith("csv") else \
     pd.read_excel(uploaded_file) if uploaded_file else load_example()

# Display input data
st.subheader("Stock Data")
st.dataframe(df)

stocks = df.iloc[:, 0]
criteria = df.columns[1:]
data = df.iloc[:, 1:].astype(float)

# Input weights
st.subheader("Input Weights (must sum to 1)")
weights = []
for i, col in enumerate(criteria):
    weight = st.number_input(f"Weight for {col}", min_value=0.0, max_value=1.0, value=round(1/len(criteria), 3), step=0.001)
    weights.append(weight)

if sum(weights) != 1:
    st.warning("Weights must sum to 1!")

# Input impact for each criterion
st.subheader("Select Impact for Each Criterion")
impact = []
for col in criteria:
    impact.append(st.selectbox(f"Impact of {col}", ["+", "-"], index=0 if "Price" not in col and "Ratio" not in col else 1))

# Step 1: Min-max normalization
st.subheader("Step 1: Min-Max Normalization")
normalized = data.copy()
for i, col in enumerate(criteria):
    x_min = data[col].min()
    x_max = data[col].max()
    if impact[i] == '+':
        normalized[col] = (data[col] - x_min) / (x_max - x_min)
    else:
        normalized[col] = (x_max - data[col]) / (x_max - x_min)
st.dataframe(normalized)

# Step 2: Weighted normalized matrix
st.subheader("Step 2: Weighted Normalized Matrix")
weighted = normalized.copy()
for i, col in enumerate(criteria):
    weighted[col] = weighted[col] * weights[i]
st.dataframe(weighted)

# Step 3: MAUT utility scores and ranking
st.subheader("Step 3: MAUT Score and Ranking")
utility = weighted.copy()
utility["MAUT_Score"] = utility.sum(axis=1)
utility["Stock"] = stocks
utility = utility[["Stock", "MAUT_Score"]]
utility = utility.sort_values(by="MAUT_Score", ascending=False).reset_index(drop=True)

# Highlight top-ranked
def highlight_top(row):
    return ['background-color: lightgreen'] * len(row) if row.name == 0 else [''] * len(row)

st.dataframe(utility.style.apply(highlight_top, axis=1))

# Chart visualization
st.subheader("MAUT Scores Visualization")
fig, ax = plt.subplots()
ax.barh(utility['Stock'], utility['MAUT_Score'], color='skyblue')
ax.set_xlabel('MAUT Score')
ax.set_title('Stock Ranking Based on MAUT Score')
st.pyplot(fig)

# CSV download
st.subheader("Download Result")
def convert_df(df): return df.to_csv(index=False).encode('utf-8')
csv = convert_df(utility)
st.download_button("Download Results as CSV", csv, "maut_stock_results.csv", "text/csv")
