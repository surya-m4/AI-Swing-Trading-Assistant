"""
Watchlists & Favorites dashboard page.

Allows users to create, manage, and view custom watchlists.
Supports adding/removing symbols and viewing live quotes
for watchlist items.
"""

import streamlit as st
from typing import Dict, List


def _get_manager():
    """Lazily imports and returns the MarketManager singleton.

    Returns:
        MarketManager instance or ``None`` on failure.
    """
    try:
        from src.market_data.market_manager import MarketManager
        return MarketManager()
    except Exception:
        return None


def render():
    """Renders the Watchlists & Favorites page."""
    st.markdown(
        "<h2 class='neon-blue'>Watchlists & Favorites</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#94A3B8;'>Create custom watchlists and manage "
        "your favorite assets. Track live prices in one place.</p>",
        unsafe_allow_html=True,
    )

    mgr = _get_manager()

    tab_fav, tab_wl, tab_create = st.tabs(
        ["⭐ Favorites", "📋 Watchlists", "➕ Create Watchlist"]
    )

    # ── Favorites tab ────────────────────────────────────────────
    with tab_fav:
        _render_favorites(mgr)

    # ── Watchlists tab ───────────────────────────────────────────
    with tab_wl:
        _render_watchlists(mgr)

    # ── Create watchlist tab ─────────────────────────────────────
    with tab_create:
        _render_create_watchlist(mgr)


def _render_favorites(mgr) -> None:
    """Renders the favorites section."""
    st.markdown(
        "<div class='glass-card fade-in'>", unsafe_allow_html=True
    )
    st.markdown("### ⭐ Your Favorites")

    if mgr is None:
        st.warning("MarketManager not available. Run the API server first.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    favorites = mgr.get_favorites()

    # Add favorite input
    col1, col2 = st.columns([3, 1])
    with col1:
        new_fav = st.text_input(
            "Add symbol to favorites",
            placeholder="e.g. RELIANCE.NS, EURUSD=X",
            key="add_fav_input",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add", key="add_fav_btn", type="primary"):
            if new_fav:
                if mgr.add_favorite(new_fav.strip().upper()):
                    st.success(f"Added {new_fav.strip().upper()}")
                    st.rerun()
                else:
                    st.info("Already in favorites.")

    if favorites:
        for i, sym in enumerate(favorites):
            col_sym, col_remove = st.columns([4, 1])
            with col_sym:
                quote = mgr.get_quote(sym)
                if quote:
                    change = quote.get("Change_Pct", 0)
                    color = "#00FF7F" if change >= 0 else "#FF3333"
                    st.markdown(
                        f"<div style='display:flex; justify-content:space-between; "
                        f"padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.1);'>"
                        f"<strong>{sym}</strong>"
                        f"<span>{quote.get('Close', 0):.2f}</span>"
                        f"<span style='color:{color}; font-weight:bold;'>"
                        f"{change:+.2f}%</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"**{sym}** — _No data_")
            with col_remove:
                if st.button("❌", key=f"rm_fav_{i}"):
                    mgr.remove_favorite(sym)
                    st.rerun()
    else:
        st.info("No favorites yet. Add symbols above to get started!")

    st.markdown("</div>", unsafe_allow_html=True)


def _render_watchlists(mgr) -> None:
    """Renders the watchlists section."""
    if mgr is None:
        st.warning("MarketManager not available.")
        return

    watchlists = mgr.get_watchlists()

    if not watchlists:
        st.info(
            "No watchlists created yet. Use the 'Create Watchlist' tab "
            "to get started."
        )
        return

    wl_name = st.selectbox(
        "Select Watchlist",
        options=list(watchlists.keys()),
        key="wl_select",
    )

    if wl_name:
        st.markdown(
            "<div class='glass-card fade-in'>", unsafe_allow_html=True
        )
        st.markdown(f"### 📋 {wl_name}")

        symbols = watchlists[wl_name]

        # Add to watchlist
        col1, col2 = st.columns([3, 1])
        with col1:
            add_sym = st.text_input(
                "Add symbol",
                placeholder="e.g. TCS.NS",
                key="wl_add_sym",
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Add", key="wl_add_btn"):
                if add_sym:
                    mgr.add_to_watchlist(wl_name, add_sym.strip().upper())
                    st.rerun()

        # Display symbols
        for i, sym in enumerate(symbols):
            col_sym, col_remove = st.columns([4, 1])
            with col_sym:
                quote = mgr.get_quote(sym)
                if quote:
                    change = quote.get("Change_Pct", 0)
                    color = "#00FF7F" if change >= 0 else "#FF3333"
                    st.markdown(
                        f"<div style='display:flex; justify-content:space-between; "
                        f"padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.1);'>"
                        f"<strong>{sym}</strong>"
                        f"<span>{quote.get('Close', 0):.2f}</span>"
                        f"<span style='color:{color}; font-weight:bold;'>"
                        f"{change:+.2f}%</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"**{sym}** — _No data_")
            with col_remove:
                if st.button("❌", key=f"rm_wl_{i}"):
                    mgr.remove_from_watchlist(wl_name, sym)
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(f"🗑️ Delete Watchlist '{wl_name}'", key="del_wl"):
            mgr.delete_watchlist(wl_name)
            st.success(f"Deleted '{wl_name}'")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def _render_create_watchlist(mgr) -> None:
    """Renders the create watchlist form."""
    st.markdown("<div class='glass-card fade-in'>", unsafe_allow_html=True)
    st.markdown("### ➕ Create New Watchlist")

    if mgr is None:
        st.warning("MarketManager not available.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    wl_name = st.text_input(
        "Watchlist Name",
        placeholder="e.g. My Tech Stocks",
        key="create_wl_name",
    )
    wl_symbols = st.text_area(
        "Symbols (one per line or comma-separated)",
        placeholder="RELIANCE.NS\nTCS.NS\nINFY.NS",
        height=150,
        key="create_wl_symbols",
    )

    if st.button("Create Watchlist", type="primary", key="create_wl_btn"):
        if wl_name and wl_symbols:
            # Parse symbols
            raw = wl_symbols.replace(",", "\n").split("\n")
            symbols = [s.strip().upper() for s in raw if s.strip()]
            mgr.create_watchlist(wl_name.strip(), symbols)
            st.success(
                f"Created watchlist '{wl_name.strip()}' with "
                f"{len(symbols)} symbols."
            )
            st.rerun()
        else:
            st.warning("Please enter both a name and at least one symbol.")

    st.markdown("</div>", unsafe_allow_html=True)
