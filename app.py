import streamlit as st
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    return pd.read_csv("final_structured_dataset.csv")

@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

df = load_data()
model = load_model()

st.title("🌍 GDP Dashboard + Predictor")

# ---------------- COUNTRY ----------------

country = st.selectbox("Select Country", df['Country Name'].unique())
filtered = df[df['Country Name'] == country]

# ---------------- KPI ----------------

st.subheader("📊 GDP Metrics")

latest = filtered['Year'].max()
prev = latest - 1

gdp_latest = filtered[filtered['Year'] == latest]['GDP'].values[0]
gdp_prev = filtered[filtered['Year'] == prev]['GDP'].values[0]

growth = ((gdp_latest - gdp_prev) / gdp_prev) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Latest GDP", f"{gdp_latest:.2f}")
col2.metric("Previous GDP", f"{gdp_prev:.2f}")
col3.metric("Growth %", f"{growth:.2f}%")

# ---------------- TREND ----------------

st.subheader("📈 GDP Trend")

temp = filtered.sort_values('Year')
temp['Year'] = temp['Year'].astype(str)

st.line_chart(temp.set_index('Year')['GDP'])

# ---------------- YEAR COMPARISON ----------------

st.subheader("📊 Compare Years")

years = sorted(filtered['Year'].unique())
y1 = st.selectbox("Year 1", years)
y2 = st.selectbox("Year 2", years, index=len(years) - 1)

v1 = filtered[filtered['Year'] == y1]['GDP'].values[0]
v2 = filtered[filtered['Year'] == y2]['GDP'].values[0]

change = ((v2 - v1) / v1) * 100

if change > 0:
    st.success(f"📈 Increase: {change:.2f}%")
else:
    st.error(f"📉 Decrease: {abs(change):.2f}%")

# ---------------- SECTOR ----------------

st.subheader("🏢 Sector Analysis")

latest_data = filtered[filtered['Year'] == latest]

scores = {
"Investment": latest_data['Investment'].values[0],
"Trade": latest_data['Trade'].values[0],
"Education": latest_data['Education'].values[0],
"Health": latest_data['LifeExp'].values[0],
"Inflation": -latest_data['Inflation'].values[0],
"Unemployment": -latest_data['Unemployment'].values[0]
}

score_df = pd.DataFrame(scores.items(), columns=["Sector", "Score"])
score_df = score_df.sort_values(by="Score", ascending=False)

st.bar_chart(score_df.set_index("Sector"))

best = score_df.iloc[0]['Sector']
worst = score_df.iloc[-1]['Sector']

st.success(f"🔥 Strong: {best}")
st.error(f"⚠ Weak: {worst}")

# ---------------- RECOMMENDATION ----------------

st.subheader("📌 Recommendation")

if worst == "Unemployment":
    st.write("Focus on job creation")
elif worst == "Inflation":
    st.write("Control inflation")
elif worst == "Investment":
    st.write("Increase investment")
elif worst == "Trade":
    st.write("Improve trade")
elif worst == "Education":
    st.write("Improve education")
elif worst == "Health":
    st.write("Improve healthcare")

# ---------------- PREDICTION ----------------

st.subheader("🤖 Predict GDP")

# UI inputs (5 inputs shown)

inflation = st.slider("Inflation (%)", 0.0, 20.0, 5.0)
unemployment = st.slider("Unemployment (%)", 0.0, 25.0, 6.0)
life_exp = st.slider("Life Expectancy", 40.0, 90.0, 70.0)
education = st.slider("Education (%)", 0.0, 100.0, 50.0)
investment = st.slider("Investment (% GDP)", 0.0, 50.0, 25.0)

# Prediction button

if st.button("🚀 Predict GDP"):
try:
# Model uses only 3 inputs
    data = np.array([[inflation, unemployment, life_exp]])
    data = sm.add_constant(data)

```
    pred = model.predict(data)

    st.success(f"💰 Predicted GDP: {pred[0]:.2f}")
except Exception as e:
    st.error(f"Error: {e}")
```

st.info("Note: Prediction uses core indicators (Inflation, Unemployment, Life Expectancy).")
