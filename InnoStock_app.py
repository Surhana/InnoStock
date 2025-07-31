# Step 3: MAUT Score and Ranking (Corrected Calculation)
st.subheader("Step 3: MAUT Score and Ranking")

# Initialize an empty list to store MAUT scores
maut_scores = []

# For each stock, calculate the weighted sum of normalized values
for index, row in normalized.iterrows():
    score = 0
    for i, col in enumerate(criteria):
        score += row[col] * weights[i]  # Multiply normalized value by the weight
    maut_scores.append(score)

# Add the MAUT scores to the dataframe
utility = pd.DataFrame({
    "Stock": stocks,
    "MAUT_Score": maut_scores
})

# Sort the dataframe based on the MAUT score
utility = utility.sort_values(by="MAUT_Score", ascending=False).reset_index(drop=True)

# Highlight top-ranked stock
def highlight_top(row):
    return ['background-color: lightgreen'] * len(row) if row.name == 0 else [''] * len(row)

st.dataframe(utility.style.apply(highlight_top, axis=1))

# Display the final MAUT Scores in a bar chart
st.subheader("MAUT Scores Visualization")
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
