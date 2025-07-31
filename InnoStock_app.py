import streamlit as st  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title and description
st.title("InnoStock: Big Data-Powered MAUT Stock Selection System")
st.markdown("""
This app evaluates and ranks stocks using **Multi-Attribute Utility Theory (MAUT)**.
Normalization uses different formulas for **benefit** and **cost** criteria based on user input.
""")

# Upload section
uploaded_file = st.file_uploader("Upload Excel or CSV file with stock data", type=["csv", "xlsx"])

# Fallback data
def load_example():
    data = {
        'Stock': ['A', 'B', 'C'],
        'Price': [100, 120, 95],
        'P/E Ratio': [15, 18, 12],
        'Dividend Yield': [2.5, 3.0, 2.8],
        'Growth Rate': [8, 7, 9]
    }
    return pd.DataFrame(data)

# Load data
df = pd.read_csv(uploaded_file) if uploaded_file and uploaded_file.name.endswith("csv") else \
     pd.read_excel(uploaded_file) if uploaded_file else load_example()

# Display uploaded/sample data
st.subheader("Stock Data")
st.dataframe(df)

# Extract data
stocks = df.iloc[:, 0]  # The first column (Stock names)
criteria = df.columns[1:]  # All columns except the first column
data = df.iloc[:, 1:].astype(float)  # All columns except the first column as numeric data

# Input weights
st.subheader("Input Weights (must sum to 1)")
weights = []
for i, col in enumerate(criteria):
    weight = st.number_input(f"Weight for {col}", min_value=0.0, max_value=1.0,
                             value=round(1/len(criteria), 3), step=0.001)
    weights.append(weight)

if round(sum(weights), 3) != 1.0:
    st.warning("⚠️ The weights must sum to 1. Please adjust.")

# Input impact — user controlled (radio buttons with unique keys)
st.subheader("Select Impact for Each Criterion")
impact = []
for i, col in enumerate(criteria):
    selected = st.radio(
        f"Is '{col}' a Benefit or Cost Criterion?",
        options=["Benefit (+)", "Cost (-)"],
        index=0,
        horizontal=True,
        key=f"impact_{i}"
    )
    impact.append("+" if "Benefit" in selected else "-")

# Step 1: Normalize using min-max scaling
st.subheader("Step 1: Min-Max Normalization")
normalized = data.copy()

for i, col in enumerate(criteria):
    x_min = data[col].min()
    x_max = data[col].max()
    if x_max == x_min:
        normalized[col] = 1  # Avoid division by 0
    elif impact[i] == '+':  # Benefit criteria
        normalized[col] = (data[col] - x_min) / (x_max - x_min)
    else:  # Cost criteria
        normalized[col] = (x_max - data[col]) / (x_max - x_min)

# ✅ Display normalized matrix
st.dataframe(normalized)

# Step 2: Weighted Normalized Matrix
st.subheader("Step 2: Weighted Normalized Matrix")
weighted = normalized.copy()
for i, col in enumerate(criteria):
    weighted[col] = weighted[col] * weights[i]

st.dataframe(weighted)

# Step 3: MAUT Score and Ranking
st.subheader("Step 3: MAUT Score and Ranking")

# Calculate MAUT Scores
maut_scores = []
for index, row in normalized.iterrows():
    score = 0
    for i, col in enumerate(criteria):
        score += row[col] * weights[i]
    maut_scores.append(score)

# Store and rank the results
utility = pd.DataFrame({
    "Stock": stocks,
    "MAUT_Score": maut_scores
})

# Sort the scores in descending order (highest first)
utility = utility.sort_values(by="MAUT_Score", ascending=False).reset_index(drop=True)

# Highlight top-ranked
def highlight_top(row):
    return ['background-color: lightgreen'] * len(row) if row.name == 0 else [''] * len(row)

st.dataframe(utility.style.apply(highlight_top, axis=1))

# Display the final MAUT Scores in a bar chart
st.subheader("Ranking the Chart")
fig, ax = plt.subplots()
ax.barh(utility['Stock'], utility['MAUT_Score'], color='skyblue')
ax.set_xlabel('MAUT Score')
ax.set_title('Stock Ranking Based on MAUT Score')
st.pyplot(fig)

# Download the results as CSV
st.subheader("Download Result")
def convert_df(df): return df.to_csv(index=False).encode('utf-8')
csv = convert_df(utility)
st.download_button("Download Results as CSV", csv, "maut_stock_results.csv", "text/csv")
