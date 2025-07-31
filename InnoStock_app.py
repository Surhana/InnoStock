import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Title and description
st.title("InnoStock: Big Data-Powered MAUT Stock Selection System")
st.markdown("""
This app allows you to evaluate and rank stocks using **Multi-Attribute Utility Theory (MAUT)**.
Upload your stock dataset, assign weights to each criterion, and the system will compute a ranking based on the utility scores.
""")

# File uploader for decision matrix (stock data)
uploaded_file = st.file_uploader("Upload Excel or CSV file with stock data", type=["csv", "xlsx"])

# Example fallback dataset
def load_example():
    data = {
        'Stock': ['A', 'B', 'C'],
        'Price': [100, 120, 95],
        'P/E Ratio': [15, 18, 12],
        'Dividend Yield': [2.5, 3.0, 2.8],
        'Growth Rate': [8, 7, 9]
    }
    return pd.DataFrame(data)

# Load the data
df = None
if uploaded_file:
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
else:
    st.info("No file uploaded. Using example dataset.")
    df = load_example()

# Display the uploaded data or example
st.subheader("Stock Data")
st.dataframe(df)

# Extract stock names and criteria
stocks = df.iloc[:, 0]
criteria = df.columns[1:]
data = df.iloc[:, 1:].astype(float)

# Input weights for each criterion
st.subheader("Input Weights (must sum to 1)")
weights = []
for i, col in enumerate(criteria):
    weight = st.number_input(f"Weight for {col}", min_value=0.0, max_value=1.0, value=1/len(criteria), step=0.001)  # Step changed to 0.001
    weights.append(weight)

# Ensure weights sum to 1
if sum(weights) != 1:
    st.warning("Weights must sum to 1! Please adjust the weights.")

# Input impact (benefit or cost for each criterion)
st.subheader("Select Impact for Each Criterion")
impact = []
for col in criteria:
    impact.append(st.selectbox(f"Impact of {col}", options=["+", "-"], index=0 if "Cost" not in col else 1))

# Normalize the data using vector normalization
st.subheader("Step 1: Normalize the Data")
normalized = data.copy()
for i, col in enumerate(criteria):
    norm = data[col] / np.sqrt((data[col]**2).sum())
    normalized[col] = norm
st.dataframe(normalized)

# Weighted Normalized Matrix
st.subheader("Step 2: Weighted Normalized Matrix")
weighted = normalized.copy()
for i, col in enumerate(criteria):
    weighted[col] = weighted[col] * weights[i]
st.dataframe(weighted)

# MAUT Utility Scores and Ranking
st.subheader("Step 3: MAUT Scores and Ranking")
utility = weighted.copy()
for i, col in enumerate(criteria):
    if impact[i] == "-":
        utility[col] = 1 - utility[col]
utility["MAUT_Score"] = utility.sum(axis=1)
utility["Stock"] = stocks
utility = utility[["Stock", "MAUT_Score"]]
utility = utility.sort_values(by="MAUT_Score", ascending=False).reset_index(drop=True)

# Highlight the top-ranked stock
def highlight_top(row):
    return ['background-color: lightgre]()
