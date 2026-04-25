import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import plotly.graph_objects as go

from execution.trader import AlpacaTrader
from data.fetcher import DataFetcher
from features.indicators import IndicatorEngine
from strategy.regime_detector import RegimeDetector
from strategy.pairs_trading import PairsTrader
from strategy.momentum import MomentumTrader
from config import settings

st.set_page_config(page_title="Equities ML Trader Live", layout="wide")

# --- Setup & Caching ---
@st.cache_resource
def get_trader():
    return AlpacaTrader()

@st.cache_resource
def get_fetcher():
    return DataFetcher()

@st.cache_data(ttl=300) # Cache for 5 minutes
def fetch_market_data(symbols, period="6mo"):
    fetcher = get_fetcher()
    data = {}
    for sym in symbols:
        df = fetcher.fetch_historical(sym, period=period)
        if not df.empty:
            data[sym] = df
    return data

@st.cache_data(ttl=300)
def detect_current_regime(_data_dict):
    if "SPY" not in _data_dict:
        return None, None, None
    df_spy = _data_dict["SPY"].copy()
    ind_engine = IndicatorEngine()
    df_spy = ind_engine.compute_all(df_spy)
    detector = RegimeDetector(n_components=getattr(settings, "REGIME_N_COMPONENTS", 3))
    detector.fit(df_spy)
    regime, conf = detector.detect_regime(df_spy)
    return regime, conf, df_spy

# --- UI Header ---
st.title("📈 Equities Mean-Reversion ML Trader")
st.markdown("Live Autonomous Execution Dashboard")

trader = get_trader()

# --- Top Row: Account Metrics ---
st.header("Live Account Status")
account = trader.get_account()
if account:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Equity", f"${account['equity']:,.2f}")
    c2.metric("Cash Balance", f"${account['cash']:,.2f}")
    c3.metric("Buying Power", f"${account['buying_power']:,.2f}")
    
    # Portfolio allocation
    allocation = (account['equity'] - account['cash']) / account['equity']
    c4.metric("Capital Deployed", f"{allocation:.1%}")
else:
    st.error("Alpaca API not connected or invalid credentials.")

st.markdown("---")

# --- Data Fetching ---
st.sidebar.header("Dashboard Controls")
st.sidebar.markdown("Fetching live market data...")
# Fetch SPY for regime, plus a handful of symbols for live visualization
viz_symbols = ["SPY"] + settings.SYMBOLS[:8] 
market_data = fetch_market_data(viz_symbols, period="1y")


# --- Section 1: Market Regime ---
st.header("1. Market Regime (Gaussian Mixture Model)")
regime, conf, df_spy_processed = detect_current_regime(market_data)

if regime is not None:
    regime_names = {
        0: "Pairs Trading (Mean-Reverting)",
        1: "Momentum (Trending)",
        2: "Cash (High Volatility / Crash)"
    }
    
    rc1, rc2 = st.columns([1, 2])
    with rc1:
        st.info(f"### Current Regime: {regime}\n**{regime_names.get(regime, 'Unknown')}**\n\nConfidence: **{conf:.1%}**")
        st.markdown("""
        The GMM calculates the hidden volatility state of the market using SPY. 
        The active regime dictates which strategy the paper trader executes today.
        """)
        
    with rc2:
        # Plot SPY and its volatility
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_spy_processed.index, y=df_spy_processed['Close'], name='SPY Close', line=dict(color='blue')))
        fig.update_layout(title="SPY Benchmark (6mo)", height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Could not calculate market regime.")

st.markdown("---")

# --- Section 2: Strategy Visualizations ---
st.header("2. Live Strategy Radars")

tab1, tab2 = st.tabs(["📉 Pairs Trading Radar", "🚀 Momentum Radar"])

with tab1:
    st.subheader("Cointegration Spread Analysis")
    st.markdown("Scanning for highly cointegrated pairs. Trades trigger when Z-Score crosses **+2.0** or **-2.0**.")
    
    if len(market_data) > 2:
        pt = PairsTrader()
        price_series = {sym: df["Close"] for sym, df in market_data.items() if sym != "SPY"}
        coint_pairs = pt.find_cointegrated_pairs(list(price_series.keys()), price_series)
        
        if coint_pairs:
            top_pair = coint_pairs[0]
            sym_a, sym_b, p_val, corr, hedge = top_pair
            
            st.success(f"**Top Cointegrated Pair:** {sym_a} & {sym_b} | p-value: {p_val:.4f} | hedge ratio: {hedge:.2f}")
            
            df_spread = pt.calculate_spread(price_series[sym_a], price_series[sym_b], hedge)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_spread.index, y=df_spread['spread_zscore'], name='Spread Z-Score', line=dict(color='purple')))
            # Add threshold lines
            fig2.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="Sell Spread")
            fig2.add_hline(y=-2.0, line_dash="dash", line_color="green", annotation_text="Buy Spread")
            fig2.add_hline(y=0, line_color="black")
            fig2.update_layout(title=f"{sym_a} / {sym_b} Spread Z-Score", height=350, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No cointegrated pairs found in the current visualization subset.")

with tab2:
    st.subheader("Trend & Momentum Ranking")
    st.markdown("Ranking stocks by multi-timeframe returns. Trades trigger if Momentum > 0.4 and ADX > 25.")
    
    mt = MomentumTrader()
    mom_scores = {}
    adx_scores = {}
    ind_engine = IndicatorEngine()
    
    for sym, df in market_data.items():
        if sym == "SPY": continue
        df_ind = ind_engine.compute_all(df.copy())
        signals = mt.generate_signals(df_ind)
        if not signals.empty:
            latest = signals.iloc[-1]
            mom_scores[sym] = latest.get("momentum_score", 0)
            adx_scores[sym] = latest.get("adx", 0)
            
    if mom_scores:
        df_mom = pd.DataFrame({"Momentum": mom_scores, "ADX": adx_scores})
        df_mom = df_mom.sort_values(by="Momentum", ascending=True)
        
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            y=df_mom.index, 
            x=df_mom["Momentum"], 
            orientation='h',
            marker=dict(color=df_mom["Momentum"], colorscale="RdYlGn", cmin=-1, cmax=1),
            name="Momentum Score"
        ))
        fig3.add_vline(x=0.4, line_dash="dash", line_color="green", annotation_text="Entry Threshold")
        fig3.update_layout(title="Current Momentum Scores", height=400, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig3, use_container_width=True)


st.markdown("---")

# --- Section 3: Open Positions & Logs ---
st.header("3. Open Positions")
positions = trader.get_positions()
if positions:
    df_pos = pd.DataFrame(positions)
    # Select useful columns
    cols = ['symbol', 'qty', 'market_value', 'unrealized_pl', 'unrealized_plpc', 'current_price', 'avg_entry_price']
    df_pos_clean = df_pos[[c for c in cols if c in df_pos.columns]].copy()
    
    # Format percentages and currencies
    if 'unrealized_plpc' in df_pos_clean:
        df_pos_clean['unrealized_plpc'] = df_pos_clean['unrealized_plpc'].astype(float).map("{:.2%}".format)
    if 'unrealized_pl' in df_pos_clean:
        df_pos_clean['unrealized_pl'] = df_pos_clean['unrealized_pl'].astype(float).map("${:,.2f}".format)
    if 'market_value' in df_pos_clean:
        df_pos_clean['market_value'] = df_pos_clean['market_value'].astype(float).map("${:,.2f}".format)
        
    st.dataframe(df_pos_clean, use_container_width=True)
else:
    st.info("No open positions. The bot is waiting for a high-probability mathematical setup.")

st.header("Recent Trade Execution Logs")
log_file = "trade_log_PORTFOLIO.csv"
if os.path.exists(log_file):
    try:
        df_trades = pd.read_csv(log_file)
        st.dataframe(df_trades.tail(15).iloc[::-1], use_container_width=True) # Show last 15, newest first
    except Exception as e:
        st.error(f"Could not load {log_file}: {e}")
else:
    st.info(f"{log_file} not found yet. It will be generated when trades are completed.")
