# # # import streamlit as st
# # # import yfinance as yf
# # # import pandas as pd
# # # import numpy as np
# # # import plotly.express as px
# # # from datetime import datetime

# # # st.set_page_config(page_title="Investment Intelligence Engine", layout="wide")

# # # st.title("💼 AI Investment Intelligence Engine")

# # # # ------------------------
# # # # Sidebar - Investor Input
# # # # ------------------------
# # # st.sidebar.header("Investor Profile")

# # # capital = st.sidebar.number_input("Investment Capital ($)", min_value=100, value=10000)
# # # risk_level = st.sidebar.selectbox("Risk Level", ["Low", "Medium", "High"])
# # # start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
# # # end_date = st.sidebar.date_input("End Date", datetime.today())

# # # # ------------------------
# # # # Stock Selection
# # # # ------------------------
# # # stocks = ["AAPL", "MSFT", "TSLA", "NVDA"]

# # # data = yf.download(stocks, start=start_date, end=end_date)["Close"]

# # # returns = data.pct_change().dropna()

# # # mean_returns = returns.mean() * 252
# # # volatility = returns.std() * np.sqrt(252)

# # # sharpe_ratio = mean_returns / volatility

# # # metrics = pd.DataFrame({
# # #     "Expected Annual Return": mean_returns,
# # #     "Annual Volatility": volatility,
# # #     "Sharpe Ratio": sharpe_ratio
# # # })

# # # st.subheader("📊 Stock Performance Metrics")
# # # st.dataframe(metrics)

# # # # ------------------------
# # # # Risk-Based Allocation
# # # # ------------------------

# # # if risk_level == "Low":
# # #     weights = np.array([0.4, 0.4, 0.1, 0.1])
# # # elif risk_level == "Medium":
# # #     weights = np.array([0.3, 0.3, 0.2, 0.2])
# # # else:
# # #     weights = np.array([0.25, 0.25, 0.25, 0.25])

# # # portfolio_return = np.sum(mean_returns * weights)
# # # portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))

# # # st.subheader("📈 Portfolio Summary")

# # # st.write("Expected Portfolio Return:", round(portfolio_return * 100, 2), "%")
# # # st.write("Expected Portfolio Volatility:", round(portfolio_volatility * 100, 2), "%")

# # # allocation_df = pd.DataFrame({
# # #     "Stock": stocks,
# # #     "Weight": weights,
# # #     "Capital Allocation ($)": weights * capital
# # # })

# # # st.subheader("💰 Capital Allocation")
# # # st.dataframe(allocation_df)

# # # fig = px.pie(allocation_df, names="Stock", values="Capital Allocation ($)", title="Portfolio Allocation")
# # # st.plotly_chart(fig, use_container_width=True)


# # import streamlit as st
# # import yfinance as yf
# # import pandas as pd
# # import numpy as np
# # import plotly.express as px
# # from datetime import datetime

# # st.set_page_config(page_title="InvestIQ - AI Investment Engine", layout="wide")
# # st.title("💼 InvestIQ - AI Investment Intelligence Engine")

# # # ------------------------
# # # Sidebar - Investor Profile
# # # ------------------------
# # st.sidebar.header("Investor Profile")

# # # Capital input
# # capital = st.sidebar.number_input("Investment Capital ($)", min_value=100, value=10000)

# # # Risk level
# # risk_level = st.sidebar.selectbox("Risk Level", ["Low", "Medium", "High"])

# # # Investment period
# # start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
# # end_date = st.sidebar.date_input("End Date", datetime.today())

# # # Number of stocks
# # st.sidebar.subheader("Portfolio Settings")
# # num_stocks = st.sidebar.slider("Number of Stocks to Invest In", min_value=1, max_value=8, value=4)

# # # ------------------------
# # # Stock Universe
# # # ------------------------
# # # US Stocks
# # us_stocks = ["AAPL", "MSFT", "TSLA", "NVDA"]

# # # NSE Stocks (Yahoo Finance uses .NS suffix)
# # nse_stocks = ["EQTY.NS", "KCB.NS", "SCOM.NS", "ABSA.NS"]

# # all_stocks = us_stocks + nse_stocks

# # # ------------------------
# # # Stock Selection
# # # ------------------------
# # selected_stocks = st.sidebar.multiselect(
# #     "Select Stocks",
# #     all_stocks,
# #     default=all_stocks[:num_stocks]
# # )

# # # Limit the selection to the number chosen
# # if len(selected_stocks) > num_stocks:
# #     st.sidebar.warning(f"Please select up to {num_stocks} stocks")
# #     selected_stocks = selected_stocks[:num_stocks]

# # if len(selected_stocks) == 0:
# #     st.warning("Please select at least one stock.")
# #     st.stop()

# # # ------------------------
# # # Fetch Data
# # # ------------------------
# # with st.spinner("Fetching stock data..."):
# #     data = yf.download(selected_stocks, start=start_date, end=end_date)["Close"]

# # # ------------------------
# # # Compute Metrics
# # # ------------------------
# # returns = data.pct_change().dropna()
# # mean_returns = returns.mean() * 252
# # volatility = returns.std() * np.sqrt(252)
# # sharpe_ratio = mean_returns / volatility

# # metrics = pd.DataFrame({
# #     "Expected Annual Return": mean_returns,
# #     "Annual Volatility": volatility,
# #     "Sharpe Ratio": sharpe_ratio
# # })

# # st.subheader("📊 Stock Performance Metrics")
# # st.dataframe(metrics.loc[selected_stocks])

# # # ------------------------
# # # Risk-Based Allocation
# # # ------------------------
# # if risk_level == "Low":
# #     weights = np.ones(len(selected_stocks)) / len(selected_stocks) * 0.8  # safer allocation
# # elif risk_level == "Medium":
# #     weights = np.ones(len(selected_stocks)) / len(selected_stocks)
# # else:
# #     weights = np.ones(len(selected_stocks)) / len(selected_stocks) * 1.2  # slightly aggressive

# # # Normalize weights to sum to 1
# # weights = weights / weights.sum()

# # portfolio_return = np.sum(mean_returns[selected_stocks] * weights)
# # portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns[selected_stocks].cov() * 252, weights)))

# # # ------------------------
# # # Display Portfolio Summary
# # # ------------------------
# # st.subheader("📈 Portfolio Summary")
# # st.write("Expected Portfolio Return:", round(portfolio_return * 100, 2), "%")
# # st.write("Expected Portfolio Volatility:", round(portfolio_volatility * 100, 2), "%")

# # # Capital allocation
# # allocation_df = pd.DataFrame({
# #     "Stock": selected_stocks,
# #     "Weight": weights,
# #     "Capital Allocation ($)": weights * capital
# # })
# # st.subheader("💰 Capital Allocation")
# # st.dataframe(allocation_df)

# # # Portfolio pie chart
# # fig = px.pie(allocation_df, names="Stock", values="Capital Allocation ($)", title="Portfolio Allocation")
# # st.plotly_chart(fig, use_container_width=True)

# # # ------------------------
# # # Recommendation Sentence
# # # ------------------------
# # top_stocks = metrics.loc[selected_stocks].sort_values("Sharpe Ratio", ascending=False).index.tolist()
# # recommendation = (
# #     f"These stocks are recommended based on historical returns and risk profile: "
# #     f"{', '.join(top_stocks)}"
# # )
# # st.info(recommendation)

# # # ------------------------
# # # Optional: Display Historical Prices Chart
# # # ------------------------
# # st.subheader("📉 Historical Prices")
# # st.line_chart(data[selected_stocks])


# """
# InvestIQ - AI Investment Intelligence Engine
# Refactored with: real risk allocation, portfolio optimization, Monte Carlo simulation,
# efficient frontier, correlation heatmap, benchmark comparison, max drawdown, caching, and more.
# """

# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from scipy.optimize import minimize
# from datetime import datetime

# # ─────────────────────────────────────────────
# # Page Config
# # ─────────────────────────────────────────────
# st.set_page_config(
#     page_title="InvestIQ - AI Investment Engine",
#     layout="wide",
#     page_icon="💼"
# )

# # ─────────────────────────────────────────────
# # Custom CSS
# # ─────────────────────────────────────────────
# st.markdown("""
# <style>
#     .metric-card {
#         background: linear-gradient(135deg, #1e293b, #0f172a);
#         border: 1px solid #334155;
#         border-radius: 12px;
#         padding: 1.2rem;
#         text-align: center;
#     }
#     .metric-value {
#         font-size: 1.8rem;
#         font-weight: 700;
#         color: #38bdf8;
#     }
#     .metric-label {
#         font-size: 0.8rem;
#         color: #94a3b8;
#         text-transform: uppercase;
#         letter-spacing: 0.05em;
#     }
#     .positive { color: #4ade80 !important; }
#     .negative { color: #f87171 !important; }
#     .section-header {
#         font-size: 1.1rem;
#         font-weight: 600;
#         color: #e2e8f0;
#         border-left: 3px solid #38bdf8;
#         padding-left: 0.75rem;
#         margin: 1.5rem 0 1rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # ─────────────────────────────────────────────
# # Constants
# # ─────────────────────────────────────────────
# STOCK_UNIVERSE = {
#     "US Large Cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM"],
#     "US ETFs":      ["SPY", "QQQ", "VTI", "GLD", "TLT", "XLE", "XLK", "IWM"],
#     "India (NSE)":  ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"],
#     "UK (LSE)":     ["BP.L", "HSBA.L", "SHEL.L", "AZN.L"],
# }
# BENCHMARK_TICKER = "SPY"
# RISK_FREE_RATE   = 0.045   # approx 4.5% annual (US T-bill, 2024)
# N_SIMULATIONS    = 1500


# # ─────────────────────────────────────────────
# # Cached Data Fetching
# # ─────────────────────────────────────────────
# @st.cache_data(show_spinner=False, ttl=3600)
# def fetch_data(tickers: list, start: str, end: str) -> pd.DataFrame:
#     """Download adjusted close prices; returns a clean DataFrame with valid tickers only."""
#     try:
#         raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
#     except Exception as e:
#         st.error(f"Download error: {e}")
#         return pd.DataFrame()

#     # Flatten MultiIndex if multiple tickers
#     if isinstance(raw.columns, pd.MultiIndex):
#         df = raw["Close"]
#     else:
#         df = raw[["Close"]] if "Close" in raw.columns else raw

#     # Drop columns that are entirely NaN (failed tickers)
#     df = df.dropna(axis=1, how="all")

#     # Forward-fill small gaps then drop any remaining NaN rows
#     df = df.ffill().dropna()
#     return df


# # ─────────────────────────────────────────────
# # Metric Helpers
# # ─────────────────────────────────────────────
# def compute_stock_metrics(prices: pd.DataFrame) -> pd.DataFrame:
#     """Annualised return, volatility, Sharpe, and max drawdown per stock."""
#     returns  = prices.pct_change().dropna()
#     ann_ret  = returns.mean() * 252
#     ann_vol  = returns.std()  * np.sqrt(252)
#     sharpe   = (ann_ret - RISK_FREE_RATE) / ann_vol

#     # Max drawdown
#     max_dd = {}
#     for col in prices.columns:
#         roll_max   = prices[col].cummax()
#         drawdown   = (prices[col] - roll_max) / roll_max
#         max_dd[col] = drawdown.min()

#     return pd.DataFrame({
#         "Ann. Return (%)":    (ann_ret  * 100).round(2),
#         "Ann. Volatility (%)": (ann_vol * 100).round(2),
#         "Sharpe Ratio":        sharpe.round(3),
#         "Max Drawdown (%)":   (pd.Series(max_dd) * 100).round(2),
#     })


# def compute_portfolio_stats(weights: np.ndarray,
#                              mean_returns: np.ndarray,
#                              cov_matrix: np.ndarray) -> tuple:
#     """Return (annualised_return, annualised_volatility, sharpe_ratio)."""
#     ret = float(np.dot(weights, mean_returns) * 252)
#     vol = float(np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(252))
#     sharpe = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0.0
#     return ret, vol, sharpe


# def portfolio_max_drawdown(weights: np.ndarray, prices: pd.DataFrame) -> float:
#     """Compute maximum drawdown for a weighted portfolio."""
#     port_value = (prices * weights).sum(axis=1)
#     port_value = port_value / port_value.iloc[0]   # normalise to 1
#     roll_max   = port_value.cummax()
#     dd         = (port_value - roll_max) / roll_max
#     return float(dd.min())


# # ─────────────────────────────────────────────
# # Risk-Based Weight Allocation
# # ─────────────────────────────────────────────
# def risk_based_weights(risk_level: str,
#                         mean_returns: pd.Series,
#                         cov_matrix: pd.DataFrame,
#                         tickers: list) -> np.ndarray:
#     """
#     Low    → Minimum Volatility optimisation
#     Medium → Maximum Sharpe optimisation
#     High   → Maximum Return (100% in highest-Sharpe stock skewed allocation)
#     """
#     n = len(tickers)
#     mr = mean_returns[tickers].values
#     cm = cov_matrix.loc[tickers, tickers].values
#     bounds = tuple((0.05, 0.60) for _ in range(n))
#     constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
#     init_w = np.ones(n) / n

#     if risk_level == "Low":
#         # Minimise portfolio volatility
#         def objective(w):
#             return np.sqrt(w @ cm @ w)
#     elif risk_level == "Medium":
#         # Maximise Sharpe ratio
#         def objective(w):
#             ret = np.dot(w, mr) * 252
#             vol = np.sqrt(w @ cm @ w) * np.sqrt(252)
#             return -(ret - RISK_FREE_RATE) / (vol + 1e-9)
#     else:
#         # Maximise return (aggressive tilt)
#         def objective(w):
#             return -np.dot(w, mr)

#     result = minimize(
#         objective, init_w,
#         method="SLSQP",
#         bounds=bounds,
#         constraints=constraints,
#         options={"maxiter": 500, "ftol": 1e-10}
#     )
#     if result.success:
#         return result.x / result.x.sum()
#     # Fallback to equal weights
#     return init_w


# # ─────────────────────────────────────────────
# # Monte Carlo Simulation
# # ─────────────────────────────────────────────
# def monte_carlo_simulation(mean_returns: pd.Series,
#                              cov_matrix: pd.DataFrame,
#                              tickers: list,
#                              n_sim: int = N_SIMULATIONS) -> pd.DataFrame:
#     """Randomly sample N portfolios and return their stats."""
#     n = len(tickers)
#     mr = mean_returns[tickers].values
#     cm = cov_matrix.loc[tickers, tickers].values
#     results = []

#     for _ in range(n_sim):
#         w = np.random.dirichlet(np.ones(n))
#         ret = float(np.dot(w, mr) * 252)
#         vol = float(np.sqrt(w @ cm @ w) * np.sqrt(252))
#         sharpe = (ret - RISK_FREE_RATE) / (vol + 1e-9)
#         results.append({"Return": ret * 100,
#                          "Volatility": vol * 100,
#                          "Sharpe": sharpe,
#                          **{t: w[i] for i, t in enumerate(tickers)}})

#     return pd.DataFrame(results)


# # ─────────────────────────────────────────────
# # Sidebar — Investor Profile
# # ─────────────────────────────────────────────
# st.sidebar.image("https://img.icons8.com/fluency/48/investment-portfolio.png", width=40)
# st.sidebar.title("InvestIQ")
# st.sidebar.markdown("---")

# st.sidebar.header("📋 Investor Profile")
# capital    = st.sidebar.number_input("Investment Capital ($)", min_value=100, value=10_000, step=500)
# risk_level = st.sidebar.selectbox("Risk Appetite", ["Low", "Medium", "High"])

# st.sidebar.markdown("---")
# st.sidebar.header("📅 Date Range")
# start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
# end_date   = st.sidebar.date_input("End Date",   datetime.today())

# if start_date >= end_date:
#     st.sidebar.error("Start date must be before end date.")
#     st.stop()

# st.sidebar.markdown("---")
# st.sidebar.header("🏦 Stock Selection")

# # Flatten stock universe for search
# all_tickers = [t for group in STOCK_UNIVERSE.values() for t in group]

# # Custom ticker input
# custom_input = st.sidebar.text_input(
#     "Add custom tickers (comma-separated)",
#     placeholder="e.g. AMZN, NFLX, BRK-B"
# )
# custom_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]

# searchable_tickers = sorted(set(all_tickers + custom_tickers))

# selected_stocks = st.sidebar.multiselect(
#     "Select Stocks (2–10)",
#     searchable_tickers,
#     default=["AAPL", "MSFT", "NVDA", "GOOGL"]
# )

# if len(selected_stocks) < 2:
#     st.warning("Please select at least 2 stocks to build a portfolio.")
#     st.stop()
# if len(selected_stocks) > 10:
#     st.warning("Maximum 10 stocks. Using the first 10.")
#     selected_stocks = selected_stocks[:10]

# include_benchmark = st.sidebar.checkbox("Include SPY Benchmark", value=True)
# show_monte_carlo  = st.sidebar.checkbox("Show Monte Carlo / Efficient Frontier", value=True)

# # ─────────────────────────────────────────────
# # Header
# # ─────────────────────────────────────────────
# st.title("💼 InvestIQ — AI Investment Intelligence Engine")
# st.caption(f"Portfolio analysis for {len(selected_stocks)} stocks · {risk_level} risk · "
#            f"{start_date.strftime('%d %b %Y')} → {end_date.strftime('%d %b %Y')}")
# st.markdown("---")

# # ─────────────────────────────────────────────
# # Fetch Data
# # ─────────────────────────────────────────────
# fetch_tickers = list(set(selected_stocks + ([BENCHMARK_TICKER] if include_benchmark else [])))

# with st.spinner("📡 Fetching market data..."):
#     prices = fetch_data(fetch_tickers, str(start_date), str(end_date))

# if prices.empty:
#     st.error("No data returned. Check your tickers and date range.")
#     st.stop()

# # Identify which selected stocks actually downloaded successfully
# valid_stocks = [t for t in selected_stocks if t in prices.columns]
# failed_stocks = [t for t in selected_stocks if t not in prices.columns]

# if failed_stocks:
#     st.warning(f"⚠️ Could not fetch data for: {', '.join(failed_stocks)}. They've been excluded.")

# if len(valid_stocks) < 2:
#     st.error("Need at least 2 valid stocks to build a portfolio.")
#     st.stop()

# # Separate benchmark
# benchmark_prices = prices[BENCHMARK_TICKER] if include_benchmark and BENCHMARK_TICKER in prices.columns else None
# stock_prices = prices[valid_stocks]

# # ─────────────────────────────────────────────
# # Compute Returns & Covariance
# # ─────────────────────────────────────────────
# daily_returns  = stock_prices.pct_change().dropna()
# mean_returns   = daily_returns.mean()
# cov_matrix     = daily_returns.cov()

# # ─────────────────────────────────────────────
# # Per-Stock Metrics
# # ─────────────────────────────────────────────
# metrics = compute_stock_metrics(stock_prices)

# st.markdown('<div class="section-header">📊 Stock Performance Metrics</div>', unsafe_allow_html=True)

# # Colour-code positive/negative returns
# def style_metrics(df):
#     def colour(val):
#         if isinstance(val, (int, float)):
#             if val > 0:
#                 return "color: #4ade80"
#             elif val < 0:
#                 return "color: #f87171"
#         return ""
#     return df.style.applymap(colour, subset=["Ann. Return (%)", "Sharpe Ratio", "Max Drawdown (%)"])

# st.dataframe(style_metrics(metrics), use_container_width=True)

# # ─────────────────────────────────────────────
# # Portfolio Optimisation
# # ─────────────────────────────────────────────
# weights = risk_based_weights(risk_level, mean_returns, cov_matrix, valid_stocks)
# p_return, p_vol, p_sharpe = compute_portfolio_stats(weights, mean_returns.values, cov_matrix.values)
# p_maxdd = portfolio_max_drawdown(weights, stock_prices)

# # ─────────────────────────────────────────────
# # Portfolio Summary Cards
# # ─────────────────────────────────────────────
# st.markdown('<div class="section-header">📈 Optimised Portfolio Summary</div>', unsafe_allow_html=True)

# c1, c2, c3, c4, c5 = st.columns(5)

# def metric_card(col, label, value, suffix="", positive_good=True):
#     colour_class = ""
#     if isinstance(value, float):
#         if positive_good and value > 0:
#             colour_class = "positive"
#         elif not positive_good and value < 0:
#             colour_class = "positive"
#         elif value < 0:
#             colour_class = "negative"
#     col.markdown(
#         f"""<div class="metric-card">
#               <div class="metric-value {colour_class}">{value:+.2f}{suffix}</div>
#               <div class="metric-label">{label}</div>
#            </div>""",
#         unsafe_allow_html=True
#     )

# metric_card(c1, "Expected Return",   p_return * 100,  "%")
# metric_card(c2, "Portfolio Volatility", p_vol * 100,   "%", positive_good=False)
# metric_card(c3, "Sharpe Ratio",       p_sharpe,         "")
# metric_card(c4, "Max Drawdown",       p_maxdd * 100,   "%", positive_good=False)
# metric_card(c5, "Total Capital",      capital,          "$")

# # ─────────────────────────────────────────────
# # Capital Allocation
# # ─────────────────────────────────────────────
# st.markdown('<div class="section-header">💰 Capital Allocation</div>', unsafe_allow_html=True)

# allocation_df = pd.DataFrame({
#     "Ticker":            valid_stocks,
#     "Weight (%)":        (weights * 100).round(2),
#     "Allocation ($)":    (weights * capital).round(2),
#     "Ann. Return (%)":   metrics["Ann. Return (%)"].values,
#     "Sharpe Ratio":      metrics["Sharpe Ratio"].values,
# }).set_index("Ticker")

# col_left, col_right = st.columns([1.2, 1])

# with col_left:
#     st.dataframe(allocation_df.style.format({
#         "Weight (%)":     "{:.2f}%",
#         "Allocation ($)": "${:,.2f}",
#     }), use_container_width=True)

# with col_right:
#     fig_pie = px.pie(
#         allocation_df.reset_index(),
#         names="Ticker",
#         values="Allocation ($)",
#         title=f"Portfolio Allocation — {risk_level} Risk",
#         color_discrete_sequence=px.colors.sequential.Blues_r,
#         hole=0.4,
#     )
#     fig_pie.update_traces(textposition="inside", textinfo="percent+label")
#     fig_pie.update_layout(
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0)",
#         font_color="#e2e8f0",
#         showlegend=False
#     )
#     st.plotly_chart(fig_pie, use_container_width=True)

# # ─────────────────────────────────────────────
# # Normalised Price Chart + Benchmark
# # ─────────────────────────────────────────────
# st.markdown('<div class="section-header">📉 Normalised Historical Performance</div>', unsafe_allow_html=True)
# st.caption("All prices re-based to 100 at the start of the period for fair comparison.")

# normalised = (stock_prices / stock_prices.iloc[0]) * 100

# if benchmark_prices is not None:
#     bench_norm = (benchmark_prices / benchmark_prices.iloc[0]) * 100
#     bench_df = bench_norm.to_frame(name="SPY (Benchmark)")
#     normalised = normalised.join(bench_df, how="left")

# fig_line = px.line(
#     normalised,
#     labels={"value": "Indexed Price (Base=100)", "variable": "Ticker"},
#     color_discrete_sequence=px.colors.qualitative.Bold,
# )
# fig_line.update_layout(
#     paper_bgcolor="rgba(0,0,0,0)",
#     plot_bgcolor="rgba(15,23,42,0.8)",
#     font_color="#e2e8f0",
#     legend_title_text="",
#     hovermode="x unified",
#     xaxis=dict(showgrid=False),
#     yaxis=dict(gridcolor="#1e293b"),
# )
# st.plotly_chart(fig_line, use_container_width=True)

# # ─────────────────────────────────────────────
# # Rolling 90-Day Returns
# # ─────────────────────────────────────────────
# st.markdown('<div class="section-header">📆 Rolling 90-Day Annualised Returns</div>', unsafe_allow_html=True)

# rolling_ret = daily_returns.rolling(90).mean() * 252 * 100
# fig_roll = px.line(
#     rolling_ret,
#     labels={"value": "90-Day Ann. Return (%)", "variable": "Ticker"},
#     color_discrete_sequence=px.colors.qualitative.Pastel,
# )
# fig_roll.add_hline(y=0, line_dash="dash", line_color="#475569")
# fig_roll.update_layout(
#     paper_bgcolor="rgba(0,0,0,0)",
#     plot_bgcolor="rgba(15,23,42,0.8)",
#     font_color="#e2e8f0",
#     hovermode="x unified",
#     xaxis=dict(showgrid=False),
#     yaxis=dict(gridcolor="#1e293b"),
# )
# st.plotly_chart(fig_roll, use_container_width=True)

# # ─────────────────────────────────────────────
# # Correlation Heatmap
# # ─────────────────────────────────────────────
# st.markdown('<div class="section-header">🔗 Return Correlation Matrix</div>', unsafe_allow_html=True)
# st.caption("Low inter-stock correlation → better diversification benefit.")

# corr = daily_returns.corr().round(2)
# fig_heat = px.imshow(
#     corr,
#     text_auto=True,
#     color_continuous_scale="RdBu_r",
#     zmin=-1, zmax=1,
#     aspect="auto",
# )
# fig_heat.update_layout(
#     paper_bgcolor="rgba(0,0,0,0)",
#     font_color="#e2e8f0",
#     coloraxis_colorbar=dict(title="Correlation"),
# )
# st.plotly_chart(fig_heat, use_container_width=True)

# # ─────────────────────────────────────────────
# # Monte Carlo / Efficient Frontier
# # ─────────────────────────────────────────────
# if show_monte_carlo:
#     st.markdown('<div class="section-header">🎲 Monte Carlo Efficient Frontier</div>', unsafe_allow_html=True)
#     st.caption(f"Plotting {N_SIMULATIONS:,} random portfolios. Your optimised portfolio is marked with ⭐.")

#     with st.spinner("Running Monte Carlo simulation..."):
#         mc_df = monte_carlo_simulation(mean_returns, cov_matrix, valid_stocks)

#     fig_mc = px.scatter(
#         mc_df,
#         x="Volatility",
#         y="Return",
#         color="Sharpe",
#         color_continuous_scale="Viridis",
#         labels={"Volatility": "Volatility (%)", "Return": "Expected Return (%)", "Sharpe": "Sharpe"},
#         opacity=0.6,
#         title="Risk vs. Return — All Simulated Portfolios",
#     )

#     # Plot optimised portfolio
#     fig_mc.add_trace(go.Scatter(
#         x=[p_vol * 100],
#         y=[p_return * 100],
#         mode="markers",
#         marker=dict(symbol="star", size=18, color="#facc15", line=dict(color="white", width=1)),
#         name="Your Portfolio",
#         showlegend=True,
#     ))

#     # Best Sharpe from simulation
#     best = mc_df.loc[mc_df["Sharpe"].idxmax()]
#     fig_mc.add_trace(go.Scatter(
#         x=[best["Volatility"]],
#         y=[best["Return"]],
#         mode="markers",
#         marker=dict(symbol="diamond", size=14, color="#f472b6", line=dict(color="white", width=1)),
#         name="Max Sharpe (Simulated)",
#         showlegend=True,
#     ))

#     fig_mc.update_layout(
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(15,23,42,0.8)",
#         font_color="#e2e8f0",
#         xaxis=dict(showgrid=False),
#         yaxis=dict(gridcolor="#1e293b"),
#         coloraxis_colorbar=dict(title="Sharpe"),
#     )
#     st.plotly_chart(fig_mc, use_container_width=True)

#     # Frontier summary
#     top5 = mc_df.nlargest(5, "Sharpe")[["Return", "Volatility", "Sharpe"] + valid_stocks]
#     st.caption("Top 5 Simulated Portfolios by Sharpe Ratio")
#     st.dataframe(top5.style.format({
#         "Return":     "{:.2f}%",
#         "Volatility": "{:.2f}%",
#         "Sharpe":     "{:.3f}",
#         **{t: "{:.1%}" for t in valid_stocks}
#     }), use_container_width=True)


# # ─────────────────────────────────────────────
# # AI Narrative Recommendation
# # ─────────────────────────────────────────────
# st.markdown('<div class="section-header">🤖 Portfolio Intelligence Report</div>', unsafe_allow_html=True)

# top_sharpe   = metrics["Sharpe Ratio"].idxmax()
# top_return   = metrics["Ann. Return (%)"].idxmax()
# lowest_vol   = metrics["Ann. Volatility (%)"].idxmin()
# worst_dd     = metrics["Max Drawdown (%)"].idxmin()
# largest_alloc = allocation_df["Weight (%)"].idxmax()

# corr_pairs = []
# for i in range(len(valid_stocks)):
#     for j in range(i + 1, len(valid_stocks)):
#         corr_pairs.append((valid_stocks[i], valid_stocks[j], corr.iloc[i, j]))
# corr_pairs.sort(key=lambda x: x[2])
# low_corr_pair = corr_pairs[0] if corr_pairs else None
# high_corr_pair = corr_pairs[-1] if corr_pairs else None

# diversification_note = ""
# if low_corr_pair:
#     diversification_note = (
#         f"The most diversifying pair is **{low_corr_pair[0]}** & **{low_corr_pair[1]}** "
#         f"(correlation: {low_corr_pair[2]:.2f}), providing meaningful risk reduction. "
#     )
# if high_corr_pair and high_corr_pair[2] > 0.85:
#     diversification_note += (
#         f"However, **{high_corr_pair[0]}** and **{high_corr_pair[1]}** are highly correlated "
#         f"({high_corr_pair[2]:.2f}), offering limited diversification benefit."
#     )

# risk_note = {
#     "Low":    "Your **Low** risk profile has been optimised to **minimise portfolio volatility**, overweighting stable, lower-beta assets.",
#     "Medium": "Your **Medium** risk profile targets the **maximum Sharpe ratio** — the best risk-adjusted return available.",
#     "High":   "Your **High** risk profile tilts toward **maximum expected returns**, accepting higher volatility for greater upside.",
# }

# st.info(f"""
# **Portfolio Intelligence Report — {risk_level} Risk Optimisation**

# {risk_note[risk_level]}

# **Standout performers:** {top_sharpe} leads on risk-adjusted return (Sharpe: {metrics.loc[top_sharpe, "Sharpe Ratio"]:.2f}), 
# while {top_return} posted the highest raw annualised return at {metrics.loc[top_return, "Ann. Return (%)"]:.1f}%. 
# The most stable holding is {lowest_vol} with {metrics.loc[lowest_vol, "Ann. Volatility (%)"]:.1f}% annual volatility.

# **Largest portfolio position:** {largest_alloc} at {allocation_df.loc[largest_alloc, "Weight (%)"]:.1f}% 
# (${allocation_df.loc[largest_alloc, "Allocation ($)"]:,.0f}).

# **Tail risk alert:** {worst_dd} experienced the steepest drawdown at {metrics.loc[worst_dd, "Max Drawdown (%)"]:.1f}%.

# **Diversification:** {diversification_note}

# *This analysis is based on historical data. Past performance does not guarantee future results. 
# Always consult a qualified financial advisor before making investment decisions.*
# """)

# # ─────────────────────────────────────────────
# # Footer
# # ─────────────────────────────────────────────
# st.markdown("---")
# st.caption(
#     "InvestIQ · Built with Streamlit & Python · "
#     "Data via Yahoo Finance (yfinance) · "
#     f"Optimisation: {'Min-Vol' if risk_level=='Low' else 'Max-Sharpe' if risk_level=='Medium' else 'Max-Return'} "
#     f"· Risk-free rate: {RISK_FREE_RATE*100:.1f}%"
# )


"""
InvestIQ - AI Investment Intelligence Engine
Refactored with session_state:
- Sidebar inputs store user selections
- Analysis runs only after "Analyse Portfolio" button click
- All features: risk allocation, portfolio optimisation, Monte Carlo, correlation heatmap, etc.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import datetime

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="InvestIQ - AI Investment Engine", layout="wide", page_icon="💼")
st.title("💼 InvestIQ - AI Investment Intelligence Engine")

# ------------------------
# Constants
# ------------------------
STOCK_UNIVERSE = {
    "US Large Cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM"],
    "US ETFs":      ["SPY", "QQQ", "VTI", "GLD", "TLT", "XLE", "XLK", "IWM"],
    "India (NSE)":  ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"],
    "UK (LSE)":     ["BP.L", "HSBA.L", "SHEL.L", "AZN.L"],
}
BENCHMARK_TICKER = "SPY"
RISK_FREE_RATE   = 0.045   # 4.5% annual
N_SIMULATIONS    = 1500

# ------------------------
# Cached Data Fetching
# ------------------------
@st.cache_data(ttl=3600)
def fetch_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw["Close"]
    else:
        df = raw[["Close"]] if "Close" in raw.columns else raw
    df = df.dropna(axis=1, how="all").ffill().dropna()
    return df

# ------------------------
# Helper Functions
# ------------------------
def compute_stock_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol
    max_dd = {col: (prices[col]/prices[col].cummax()-1).min() for col in prices.columns}
    return pd.DataFrame({
        "Ann. Return (%)":    (ann_ret*100).round(2),
        "Ann. Volatility (%)": (ann_vol*100).round(2),
        "Sharpe Ratio":       sharpe.round(3),
        "Max Drawdown (%)":   (pd.Series(max_dd)*100).round(2)
    })

def compute_portfolio_stats(weights, mean_returns, cov_matrix):
    ret = float(np.dot(weights, mean_returns) * 252)
    vol = float(np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(252))
    sharpe = (ret - RISK_FREE_RATE)/vol if vol>0 else 0
    return ret, vol, sharpe

def portfolio_max_drawdown(weights, prices):
    port_value = (prices*weights).sum(axis=1)
    port_value /= port_value.iloc[0]
    dd = (port_value.cummax() - port_value)/port_value.cummax()
    return float(dd.max())

def risk_based_weights(risk_level, mean_returns, cov_matrix, tickers):
    n = len(tickers)
    mr = mean_returns[tickers].values
    cm = cov_matrix.loc[tickers, tickers].values
    bounds = tuple((0.05,0.6) for _ in range(n))
    constraints = [{"type":"eq","fun":lambda w: np.sum(w)-1}]
    init_w = np.ones(n)/n

    if risk_level=="Low":
        objective = lambda w: np.sqrt(w @ cm @ w)
    elif risk_level=="Medium":
        objective = lambda w: -(np.dot(w,mr)*252 - RISK_FREE_RATE)/np.sqrt(w @ cm @ w * 252 + 1e-9)
    else:  # High
        objective = lambda w: -np.dot(w,mr)

    res = minimize(objective, init_w, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter":500,"ftol":1e-10})
    if res.success:
        return res.x/res.x.sum()
    return init_w

def monte_carlo_simulation(mean_returns, cov_matrix, tickers, n_sim=N_SIMULATIONS):
    n = len(tickers)
    mr = mean_returns[tickers].values
    cm = cov_matrix.loc[tickers, tickers].values
    results = []
    for _ in range(n_sim):
        w = np.random.dirichlet(np.ones(n))
        ret = float(np.dot(w,mr)*252)
        vol = float(np.sqrt(w @ cm @ w) * np.sqrt(252))
        sharpe = (ret-RISK_FREE_RATE)/ (vol + 1e-9)
        results.append({"Return":ret*100,"Volatility":vol*100,"Sharpe":sharpe,**{t:w[i] for i,t in enumerate(tickers)}})
    return pd.DataFrame(results)

# ------------------------
# Session State Defaults
# ------------------------
if "capital" not in st.session_state: st.session_state.capital = 10000
if "risk_level" not in st.session_state: st.session_state.risk_level = "Medium"
if "start_date" not in st.session_state: st.session_state.start_date = pd.to_datetime("2020-01-01")
if "end_date" not in st.session_state: st.session_state.end_date = datetime.today()
if "selected_stocks" not in st.session_state: st.session_state.selected_stocks = ["AAPL","MSFT","NVDA","GOOGL"]
if "include_benchmark" not in st.session_state: st.session_state.include_benchmark = True
if "show_monte_carlo" not in st.session_state: st.session_state.show_monte_carlo = True

# ------------------------
# Sidebar Inputs
# ------------------------
st.sidebar.header("📋 Investor Profile")
st.session_state.capital = st.sidebar.number_input("Investment Capital ($)", min_value=100,
                                                    value=st.session_state.capital, step=500)
st.session_state.risk_level = st.sidebar.selectbox("Risk Appetite", ["Low","Medium","High"],
                                                   index=["Low","Medium","High"].index(st.session_state.risk_level))

st.sidebar.header("📅 Date Range")
st.session_state.start_date = st.sidebar.date_input("Start Date", st.session_state.start_date)
st.session_state.end_date   = st.sidebar.date_input("End Date", st.session_state.end_date)

st.sidebar.header("🏦 Stock Selection")
custom_input = st.sidebar.text_input("Add custom tickers (comma-separated)", placeholder="e.g. AMZN, NFLX")
custom_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
all_tickers = [t for group in STOCK_UNIVERSE.values() for t in group]
searchable_tickers = sorted(set(all_tickers + custom_tickers))

st.session_state.selected_stocks = st.sidebar.multiselect("Select Stocks (2–10)",
                                                          searchable_tickers,
                                                          default=st.session_state.selected_stocks)

st.session_state.include_benchmark = st.sidebar.checkbox("Include SPY Benchmark",
                                                         value=st.session_state.include_benchmark)
st.session_state.show_monte_carlo = st.sidebar.checkbox("Show Monte Carlo / Efficient Frontier",
                                                        value=st.session_state.show_monte_carlo)

# ------------------------
# Analyse Portfolio Button
# ------------------------
analyse_clicked = st.sidebar.button("🔍 Analyse Portfolio")

# ------------------------
# Run Analysis Only When Clicked
# ------------------------
if analyse_clicked:

    # Assign variables from session_state
    capital = st.session_state.capital
    risk_level = st.session_state.risk_level
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    selected_stocks = st.session_state.selected_stocks
    include_benchmark = st.session_state.include_benchmark
    show_monte_carlo = st.session_state.show_monte_carlo

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()
    if len(selected_stocks)<2:
        st.error("Select at least 2 stocks to analyse.")
        st.stop()
    if len(selected_stocks)>10:
        st.warning("Maximum 10 stocks. Using first 10.")
        selected_stocks = selected_stocks[:10]

    # Fetch Data
    fetch_tickers = list(set(selected_stocks + ([BENCHMARK_TICKER] if include_benchmark else [])))
    with st.spinner("📡 Fetching market data..."):
        prices = fetch_data(fetch_tickers, str(start_date), str(end_date))

    valid_stocks = [t for t in selected_stocks if t in prices.columns]
    failed_stocks = [t for t in selected_stocks if t not in prices.columns]
    if failed_stocks:
        st.warning(f"⚠️ Could not fetch data for: {', '.join(failed_stocks)}")

    if len(valid_stocks)<2:
        st.error("Need at least 2 valid stocks.")
        st.stop()

    benchmark_prices = prices[BENCHMARK_TICKER] if include_benchmark and BENCHMARK_TICKER in prices.columns else None
    stock_prices = prices[valid_stocks]

    # Compute Returns & Covariance
    daily_returns = stock_prices.pct_change().dropna()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    # Per-stock Metrics
    metrics = compute_stock_metrics(stock_prices)
    st.subheader("📊 Stock Metrics")
    st.dataframe(metrics, use_container_width=True)

    # Portfolio Optimisation
    weights = risk_based_weights(risk_level, mean_returns, cov_matrix, valid_stocks)
    p_return, p_vol, p_sharpe = compute_portfolio_stats(weights, mean_returns.values, cov_matrix.values)
    p_maxdd = portfolio_max_drawdown(weights, stock_prices)

    # Portfolio Allocation
    allocation_df = pd.DataFrame({
        "Ticker": valid_stocks,
        "Weight (%)": (weights*100).round(2),
        "Allocation ($)": (weights*capital).round(2),
        "Ann. Return (%)": metrics["Ann. Return (%)"].values,
        "Sharpe Ratio": metrics["Sharpe Ratio"].values
    }).set_index("Ticker")
    st.subheader("💰 Portfolio Allocation")
    st.dataframe(allocation_df, use_container_width=True)

    # Normalised Price Chart
    st.subheader("📉 Normalised Historical Performance")
    normalised = (stock_prices/stock_prices.iloc[0])*100
    if benchmark_prices is not None:
        bench_norm = (benchmark_prices/benchmark_prices.iloc[0])*100
        normalised = normalised.join(bench_norm.to_frame(name="SPY (Benchmark)"))
    st.line_chart(normalised)

    # Monte Carlo Efficient Frontier
    if show_monte_carlo:
        st.subheader("🎲 Monte Carlo / Efficient Frontier")
        with st.spinner("Running Monte Carlo simulation..."):
            mc_df = monte_carlo_simulation(mean_returns, cov_matrix, valid_stocks)
        fig_mc = px.scatter(mc_df, x="Volatility", y="Return", color="Sharpe",
                            color_continuous_scale="Viridis", opacity=0.6)
        # Add user portfolio
        fig_mc.add_trace(go.Scatter(x=[p_vol*100], y=[p_return*100],
                                    mode="markers", marker=dict(symbol="star", size=18, color="gold"),
                                    name="Your Portfolio"))
        st.plotly_chart(fig_mc, use_container_width=True)

    # AI Recommendation
    st.subheader("🤖 Portfolio Intelligence Report")
    top_sharpe = metrics["Sharpe Ratio"].idxmax()
    top_return = metrics["Ann. Return (%)"].idxmax()
    lowest_vol = metrics["Ann. Volatility (%)"].idxmin()
    worst_dd = metrics["Max Drawdown (%)"].idxmin()
    largest_alloc = allocation_df["Weight (%)"].idxmax()

    st.info(f"""
**Portfolio Intelligence Report — {risk_level} Risk Optimisation**

Standout performers: {top_sharpe} (Sharpe: {metrics.loc[top_sharpe,'Sharpe Ratio']:.2f}),
{top_return} (Annual Return: {metrics.loc[top_return,'Ann. Return (%)']:.1f}%).
Most stable: {lowest_vol} ({metrics.loc[lowest_vol,'Ann. Volatility (%)']:.1f}% vol).
Largest allocation: {largest_alloc} ({allocation_df.loc[largest_alloc,'Weight (%)']:.1f}% - ${allocation_df.loc[largest_alloc,'Allocation ($)']:,.0f})
Steepest drawdown: {worst_dd} ({metrics.loc[worst_dd,'Max Drawdown (%)']:.1f}%)
""")
