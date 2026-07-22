"""
Streamlit Portfolio & Paper Trading Dashboard Page for Module 14.

Features a professional dark-themed trading platform interface with modern cards,
gradient buttons, rounded corners, interactive Buy/Sell panels, holdings table,
searchable trade history with CSV export, and Plotly performance analytics charts.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, List


def _get_trader():
    """Instantiates or returns PaperTrader singleton."""
    try:
        from src.trading.paper_trader import PaperTrader
        return PaperTrader()
    except Exception as exc:
        st.error(f"Failed to load PaperTrader module: {exc}")
        return None


def render():
    """Renders the dark-themed Paper Trading dashboard page."""
    st.markdown("<h2 class='neon-blue'>Paper Trading & Portfolio Analytics</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94A3B8;'>Professional AI swing trading platform. "
        "Simulate trading Indian stocks and Forex with virtual capital ₹10,00,000.</p>",
        unsafe_allow_html=True,
    )

    trader = _get_trader()
    if trader is None:
        return

    # Fetch live summary and analytics
    summary = trader.get_portfolio()
    analytics = trader.get_analytics()

    # ── Summary Cards Row 1 ───────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
            <div class="glass-card fade-in" style="border-radius: 12px; background: rgba(15, 23, 42, 0.7); padding: 15px;">
                <div style="color: #94A3B8; font-size: 0.8rem; font-weight: 600;">PORTFOLIO VALUE</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #00C8FF;">₹ {summary.portfolio_value:,.2f}</div>
                <div style="color: #94A3B8; font-size: 0.75rem;">Initial: ₹ {trader.portfolio.initial_capital:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="glass-card fade-in" style="border-radius: 12px; background: rgba(15, 23, 42, 0.7); padding: 15px;">
                <div style="color: #94A3B8; font-size: 0.8rem; font-weight: 600;">AVAILABLE CASH</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #FFF;">₹ {summary.cash:,.2f}</div>
                <div style="color: #94A3B8; font-size: 0.75rem;">Margin: ₹ {summary.available_margin:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="glass-card fade-in" style="border-radius: 12px; background: rgba(15, 23, 42, 0.7); padding: 15px;">
                <div style="color: #94A3B8; font-size: 0.8rem; font-weight: 600;">INVESTED AMOUNT</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #8A2BE2;">₹ {summary.invested_amount:,.2f}</div>
                <div style="color: #94A3B8; font-size: 0.75rem;">Active Holdings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        pnl_color = "#00FF7F" if summary.overall_pnl >= 0 else "#FF3333"
        st.markdown(
            f"""
            <div class="glass-card fade-in" style="border-radius: 12px; background: rgba(15, 23, 42, 0.7); padding: 15px;">
                <div style="color: #94A3B8; font-size: 0.8rem; font-weight: 600;">OVERALL P&L</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: {pnl_color};">₹ {summary.overall_pnl:+,.2f}</div>
                <div style="color: {pnl_color}; font-size: 0.75rem; font-weight: bold;">ROI: {summary.roi:+.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
            <div class="glass-card fade-in" style="border-radius: 12px; background: rgba(15, 23, 42, 0.7); padding: 15px;">
                <div style="color: #94A3B8; font-size: 0.8rem; font-weight: 600;">WIN RATE</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #00FF7F;">{summary.win_rate:.1f}%</div>
                <div style="color: #FF3333; font-size: 0.75rem;">Max DD: {summary.max_drawdown:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs Navigation ───────────────────────────────────────────────
    tab_charts, tab_buy, tab_sell, tab_pos, tab_history, tab_risk = st.tabs(
        [
            "📈 Performance Charts",
            "🟢 Buy Panel",
            "🔴 Sell Panel",
            "📊 Current Holdings",
            "📜 Trade History",
            "🛡️ Risk & Analytics",
        ]
    )

    # ── Tab 1: Performance Charts ────────────────────────────────────
    with tab_charts:
        c_left, c_right = st.columns([2, 1])

        with c_left:
            st.markdown("<div class='glass-card fade-in'>", unsafe_allow_html=True)
            st.markdown("### Daily Equity Curve")

            snapshots = trader.portfolio.equity_snapshots
            fig_equity = go.Figure()
            fig_equity.add_trace(
                go.Scatter(
                    y=snapshots,
                    mode="lines+markers",
                    name="Portfolio Equity (₹)",
                    line=dict(color="#00C8FF", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(0, 200, 255, 0.1)",
                )
            )
            fig_equity.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=20, l=20, r=20),
                height=300,
                xaxis_title="Execution Tick",
                yaxis_title="Equity (₹)",
            )
            st.plotly_chart(fig_equity, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_right:
            st.markdown("<div class='glass-card fade-in'>", unsafe_allow_html=True)
            st.markdown("### Asset Allocation")

            labels = ["Available Cash"]
            values = [summary.cash]
            colors = ["#00FF7F"]

            positions = trader.get_positions()
            for p in positions:
                labels.append(p["ticker"])
                values.append(p["current_value"])

            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.45,
                        textinfo="label+percent",
                    )
                ]
            )
            fig_pie.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=20, l=20, r=20),
                height=300,
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 2: Buy Panel ─────────────────────────────────────────────
    with tab_buy:
        st.markdown("<div class='glass-card fade-in'>", unsafe_allow_html=True)
        st.markdown("### 🟢 Execute BUY Order")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            buy_ticker = st.text_input("Ticker Symbol", value="RELIANCE.NS", key="b_ticker").strip().upper()
            buy_qty = st.number_input("Quantity", min_value=1.0, value=10.0, step=1.0, key="b_qty")
            buy_price = st.number_input("Execution Price (₹)", min_value=0.01, value=2500.0, step=1.0, key="b_price")

        with col_b2:
            buy_sl = st.number_input("Stop Loss (Optional)", value=2425.0, step=1.0, key="b_sl")
            buy_tp = st.number_input("Take Profit (Optional)", value=2650.0, step=1.0, key="b_tp")
            buy_ts = st.number_input("Trailing Stop Offset (Optional)", value=0.0, step=1.0, key="b_ts")

        est_cost = buy_qty * buy_price
        st.caption(f"Estimated Order Cost: **₹{est_cost:,.2f}**  |  Available Cash: **₹{summary.cash:,.2f}**")

        if st.button("🚀 Place BUY Order", type="primary", key="btn_exec_buy"):
            try:
                sl_val = buy_sl if buy_sl > 0 else None
                tp_val = buy_tp if buy_tp > 0 else None
                ts_val = buy_ts if buy_ts > 0 else None

                resp = trader.buy(
                    ticker=buy_ticker,
                    quantity=buy_qty,
                    price=buy_price,
                    stop_loss=sl_val,
                    take_profit=tp_val,
                    trailing_stop=ts_val,
                )
                st.success(f"Order Executed! Trade ID: {resp.trade_id} — {resp.message}")
                st.rerun()
            except Exception as exc:
                st.error(f"BUY Order Rejected: {exc}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 3: Sell Panel ────────────────────────────────────────────
    with tab_sell:
        st.markdown("<div class='glass-card fade-in'>", unsafe_allow_html=True)
        st.markdown("### 🔴 Execute SELL Order (Partial or Full Close)")

        positions = trader.get_positions()
        if not positions:
            st.info("No open holdings to sell.")
        else:
            ticker_list = [p["ticker"] for p in positions]
            sell_ticker = st.selectbox("Select Holding to Sell", options=ticker_list, key="s_ticker")

            pos_obj = next((p for p in positions if p["ticker"] == sell_ticker), None)
            if pos_obj:
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.caption(f"Max Owned Quantity: **{pos_obj['quantity']}**  |  Avg Entry Price: **₹{pos_obj['entry_price']:,.2f}**")
                    sell_qty = st.number_input(
                        "Sell Quantity",
                        min_value=0.01,
                        max_value=float(pos_obj["quantity"]),
                        value=float(pos_obj["quantity"]),
                        key="s_qty",
                    )
                with col_s2:
                    sell_price = st.number_input("Exit Price (₹)", min_value=0.01, value=float(pos_obj["current_price"]), key="s_price")

                if st.button("🔥 Execute SELL Order", type="primary", key="btn_exec_sell"):
                    try:
                        resp = trader.sell(ticker=sell_ticker, quantity=sell_qty, price=sell_price)
                        st.success(f"Sell Order Executed! {resp.message}")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"SELL Order Failed: {exc}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 4: Current Holdings ──────────────────────────────────────
    with tab_pos:
        st.markdown("<div class='glass-card fade-in'>", unsafe_allow_html=True)
        st.markdown("### 📊 Active Positions")

        positions = trader.get_positions()
        if not positions:
            st.info("No open positions currently held.")
        else:
            table_html = (
                "<table class='dataframe'>"
                "<tr><th>Ticker</th><th>Entry</th><th>Current</th><th>Qty</th>"
                "<th>Value</th><th>P&L</th><th>Return %</th><th>Hold Days</th><th>SL</th><th>TP</th><th>Action</th></tr>"
            )
            for p in positions:
                pnl_color = "#00FF7F" if p["pnl"] >= 0 else "#FF3333"
                sl_str = f"₹{p['stop_loss']:.2f}" if p.get('stop_loss') else "-"
                tp_str = f"₹{p['take_profit']:.2f}" if p.get('take_profit') else "-"

                table_html += (
                    f"<tr>"
                    f"<td><strong>{p['ticker']}</strong></td>"
                    f"<td>₹{p['entry_price']:,.2f}</td>"
                    f"<td>₹{p['current_price']:,.2f}</td>"
                    f"<td>{p['quantity']}</td>"
                    f"<td>₹{p['current_value']:,.2f}</td>"
                    f"<td style='color:{pnl_color}; font-weight:bold;'>₹{p['pnl']:+,.2f}</td>"
                    f"<td style='color:{pnl_color}; font-weight:bold;'>{p['pnl_pct']:+.2f}%</td>"
                    f"<td>{p['holding_days']}d</td>"
                    f"<td>{sl_str}</td>"
                    f"<td>{tp_str}</td>"
                    f"<td><span style='color:#FF3333; font-weight:bold;'>Active</span></td>"
                    f"</tr>"
                )
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 5: Trade History & CSV Export ────────────────────────────
    with tab_history:
        st.markdown("<div class='glass-card fade-in'>", unsafe_allow_html=True)
        col_h1, col_h2 = st.columns([3, 1])

        with col_h1:
            st.markdown("### 📜 Trade Log")
        with col_h2:
            if st.button("📥 Export CSV", key="btn_export_csv"):
                csv_path = trader.export_history_csv()
                st.success(f"Exported to {csv_path}")

        trades = trader.get_trade_history()
        if not trades:
            st.info("No transaction history recorded.")
        else:
            df_trades = pd.DataFrame(trades)
            st.dataframe(df_trades, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 6: Risk & Analytics ──────────────────────────────────────
    with tab_risk:
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.markdown("<div class='glass-card fade-in'>", unsafe_allow_html=True)
            st.markdown("### 🎯 Quantitative Analytics")

            st.markdown(f"- **Portfolio Return**: `{analytics.portfolio_return:+.2f}%`")
            st.markdown(f"- **Sharpe Ratio**: `{analytics.sharpe_ratio:.2f}`")
            st.markdown(f"- **Sortino Ratio**: `{analytics.sortino_ratio:.2f}`")
            st.markdown(f"- **Profit Factor**: `{analytics.profit_factor:.2f}`")
            st.markdown(f"- **Win Percentage**: `{analytics.win_percentage:.1f}%`")
            st.markdown(f"- **Average Winning Trade**: `₹{analytics.avg_winning_trade:,.2f}`")
            st.markdown(f"- **Average Losing Trade**: `₹{analytics.avg_losing_trade:,.2f}`")
            st.markdown(f"- **Total Profit**: `₹{analytics.total_profit:,.2f}`")
            st.markdown(f"- **Total Loss**: `₹{analytics.total_loss:,.2f}`")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_r2:
            st.markdown("<div class='glass-card fade-in'>", unsafe_allow_html=True)
            st.markdown("### 🛡️ Pre-Trade Risk Controls")

            risk_st = trader.get_risk_status()
            st.markdown(f"- **Max Position Size Cap**: `{risk_st.max_position_size_pct:.1f}% of Portfolio`")
            st.markdown(f"- **Available Cash**: `₹{risk_st.cash_available:,.2f}`")
            st.markdown(f"- **Simulated Market Open**: `{risk_st.is_market_open}`")
            st.markdown(f"- **Risk Control Status**: `{risk_st.status_message}`")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⚠️ Reset Virtual Account (₹10,00,000)", key="btn_reset_trading_acc"):
                trader.reset_account(1000000.0)
                st.success("Account reset to ₹10,00,000!")
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)
