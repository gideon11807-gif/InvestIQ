import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Investment Intelligence Engine", layout="wide")

st.title("💼 AI Investment Intelligence Engine")

# ------------------------
# Sidebar - Investor Input
# ------------------------
st.sidebar.header("Investor Profile")

capital = st.sidebar.number_input("Investment Capital ($)", min_value=100, value=10000)
risk_level = st.sidebar.selectbox("Risk Level", ["Low", "Medium", "High"])
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", datetime.today())

# ------------------------
# Stock Selection
# ------------------------
stocks = ["AAPL", "MSFT", "TSLA", "NVDA"]

data = yf.download(stocks, start=start_date, end=end_date)["Close"]

returns = data.pct_change().dropna()

mean_returns = returns.mean() * 252
volatility = returns.std() * np.sqrt(252)

sharpe_ratio = mean_returns / volatility

metrics = pd.DataFrame({
    "Expected Annual Return": mean_returns,
    "Annual Volatility": volatility,
    "Sharpe Ratio": sharpe_ratio
})

st.subheader("📊 Stock Performance Metrics")
st.dataframe(metrics)

# ------------------------
# Risk-Based Allocation
# ------------------------

if risk_level == "Low":
    weights = np.array([0.4, 0.4, 0.1, 0.1])
elif risk_level == "Medium":
    weights = np.array([0.3, 0.3, 0.2, 0.2])
else:
    weights = np.array([0.25, 0.25, 0.25, 0.25])

portfolio_return = np.sum(mean_returns * weights)
portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))

st.subheader("📈 Portfolio Summary")

st.write("Expected Portfolio Return:", round(portfolio_return * 100, 2), "%")
st.write("Expected Portfolio Volatility:", round(portfolio_volatility * 100, 2), "%")

allocation_df = pd.DataFrame({
    "Stock": stocks,
    "Weight": weights,
    "Capital Allocation ($)": weights * capital
})

st.subheader("💰 Capital Allocation")
st.dataframe(allocation_df)

fig = px.pie(allocation_df, names="Stock", values="Capital Allocation ($)", title="Portfolio Allocation")
st.plotly_chart(fig, use_container_width=True)
