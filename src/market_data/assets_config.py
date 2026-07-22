"""
Centralised asset configuration for the AI Swing Trading Assistant.

All tradable symbols are defined here and loaded dynamically at runtime.
Dashboard pages and API endpoints read from ``AssetRegistry`` instead of
hard-coding ticker lists.

Categories
----------
- INDIAN_STOCKS  – NSE-listed equities (100+ large / mid / small cap)
- FOREX_MAJOR    – Seven most-traded G10 pairs
- FOREX_MINOR    – Cross-currency pairs without USD
- FOREX_INR      – Rupee-denominated pairs
- INDICES        – Benchmark index tickers
- COMMODITIES    – Precious metals, energy, agriculture
- CRYPTO         – Major cryptocurrency pairs
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ── Category enum ────────────────────────────────────────────────────


class AssetCategory(str, enum.Enum):
    """Enumeration of supported asset categories."""

    INDIAN_STOCKS = "Indian Stocks"
    FOREX_MAJOR = "Forex Major"
    FOREX_MINOR = "Forex Minor"
    FOREX_INR = "Forex INR"
    INDICES = "Indices"
    COMMODITIES = "Commodities"
    CRYPTO = "Crypto"


# ── Asset dataclass ──────────────────────────────────────────────────


@dataclass(frozen=True)
class AssetInfo:
    """Immutable metadata for a single tradeable asset.

    Attributes:
        symbol: Provider-specific ticker (e.g. ``RELIANCE.NS``).
        name: Human-readable display name.
        category: Asset category for filtering.
        exchange: Exchange or provider name.
    """

    symbol: str
    name: str
    category: AssetCategory
    exchange: str = ""


# ── Static catalogue ─────────────────────────────────────────────────

# Indian Stocks — NSE (.NS suffix for yfinance)
INDIAN_STOCKS: List[AssetInfo] = [
    # ── Large-Cap Blue Chips ──
    AssetInfo("RELIANCE.NS", "Reliance Industries", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("TCS.NS", "Tata Consultancy Services", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("INFY.NS", "Infosys", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("HDFCBANK.NS", "HDFC Bank", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("ICICIBANK.NS", "ICICI Bank", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("SBIN.NS", "State Bank of India", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("HINDUNILVR.NS", "Hindustan Unilever", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("BHARTIARTL.NS", "Bharti Airtel", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("ITC.NS", "ITC Limited", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("KOTAKBANK.NS", "Kotak Mahindra Bank", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("LT.NS", "Larsen & Toubro", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("HCLTECH.NS", "HCL Technologies", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("ASIANPAINT.NS", "Asian Paints", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("AXISBANK.NS", "Axis Bank", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("MARUTI.NS", "Maruti Suzuki", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("SUNPHARMA.NS", "Sun Pharmaceutical", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("TITAN.NS", "Titan Company", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("BAJFINANCE.NS", "Bajaj Finance", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("WIPRO.NS", "Wipro", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("ULTRACEMCO.NS", "UltraTech Cement", AssetCategory.INDIAN_STOCKS, "NSE"),
    # ── Large-Cap (continued) ──
    AssetInfo("NESTLEIND.NS", "Nestle India", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("NTPC.NS", "NTPC Limited", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("TATAMOTORS.NS", "Tata Motors", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("TATASTEEL.NS", "Tata Steel", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("POWERGRID.NS", "Power Grid Corp", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("M&M.NS", "Mahindra & Mahindra", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("JSWSTEEL.NS", "JSW Steel", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("ADANIENT.NS", "Adani Enterprises", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("ADANIPORTS.NS", "Adani Ports", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("ONGC.NS", "ONGC", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("COALINDIA.NS", "Coal India", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("BPCL.NS", "Bharat Petroleum", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("IOC.NS", "Indian Oil Corp", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("GRASIM.NS", "Grasim Industries", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("TECHM.NS", "Tech Mahindra", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("DRREDDY.NS", "Dr. Reddy's Labs", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("CIPLA.NS", "Cipla", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("HEROMOTOCO.NS", "Hero MotoCorp", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("EICHERMOT.NS", "Eicher Motors", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("DIVISLAB.NS", "Divi's Labs", AssetCategory.INDIAN_STOCKS, "NSE"),
    # ── Mid-Cap ──
    AssetInfo("BAJAJFINSV.NS", "Bajaj Finserv", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("BRITANNIA.NS", "Britannia Industries", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("APOLLOHOSP.NS", "Apollo Hospitals", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("INDUSINDBK.NS", "IndusInd Bank", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("SBILIFE.NS", "SBI Life Insurance", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("HDFCLIFE.NS", "HDFC Life Insurance", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("SHREECEM.NS", "Shree Cement", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("DABUR.NS", "Dabur India", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("PIDILITIND.NS", "Pidilite Industries", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("HAVELLS.NS", "Havells India", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("GODREJCP.NS", "Godrej Consumer", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("SIEMENS.NS", "Siemens India", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("AMBUJACEM.NS", "Ambuja Cements", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("ACC.NS", "ACC Limited", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("BERGEPAINT.NS", "Berger Paints", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("HINDPETRO.NS", "Hindustan Petroleum", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("BANKBARODA.NS", "Bank of Baroda", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("PNB.NS", "Punjab National Bank", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("CANBK.NS", "Canara Bank", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("FEDERALBNK.NS", "Federal Bank", AssetCategory.INDIAN_STOCKS, "NSE"),
    # ── IT / Tech ──
    AssetInfo("LTIM.NS", "LTIMindtree", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("MPHASIS.NS", "Mphasis", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("PERSISTENT.NS", "Persistent Systems", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("COFORGE.NS", "Coforge", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("LTTS.NS", "L&T Technology Services", AssetCategory.INDIAN_STOCKS, "NSE"),
    # ── Pharma / Healthcare ──
    AssetInfo("AUROPHARMA.NS", "Aurobindo Pharma", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("BIOCON.NS", "Biocon", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("LUPIN.NS", "Lupin", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("TORNTPHARM.NS", "Torrent Pharmaceuticals", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("ALKEM.NS", "Alkem Labs", AssetCategory.INDIAN_STOCKS, "NSE"),
    # ── Auto ──
    AssetInfo("BAJAJ-AUTO.NS", "Bajaj Auto", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("TVSMOTOR.NS", "TVS Motor", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("ASHOKLEY.NS", "Ashok Leyland", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("MOTHERSON.NS", "Motherson Sumi", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("BALKRISIND.NS", "Balkrishna Industries", AssetCategory.INDIAN_STOCKS, "NSE"),
    # ── FMCG / Consumer ──
    AssetInfo("MARICO.NS", "Marico", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("COLPAL.NS", "Colgate-Palmolive India", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("TATACONSUM.NS", "Tata Consumer Products", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("UBL.NS", "United Breweries", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("JUBLFOOD.NS", "Jubilant FoodWorks", AssetCategory.INDIAN_STOCKS, "NSE"),
    # ── Energy / Infra ──
    AssetInfo("ADANIGREEN.NS", "Adani Green Energy", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("TATAPOWER.NS", "Tata Power", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("NHPC.NS", "NHPC", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("IRCTC.NS", "IRCTC", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("HAL.NS", "Hindustan Aeronautics", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("BEL.NS", "Bharat Electronics", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("BHEL.NS", "Bharat Heavy Electricals", AssetCategory.INDIAN_STOCKS, "NSE"),
    # ── Metals / Mining ──
    AssetInfo("HINDALCO.NS", "Hindalco Industries", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("VEDL.NS", "Vedanta", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("NMDC.NS", "NMDC", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("SAIL.NS", "Steel Authority of India", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("NATIONALUM.NS", "National Aluminium", AssetCategory.INDIAN_STOCKS, "NSE"),
    # ── Financials ──
    AssetInfo("ICICIGI.NS", "ICICI Lombard", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("MUTHOOTFIN.NS", "Muthoot Finance", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("CHOLAFIN.NS", "Cholamandalam Investment", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("MANAPPURAM.NS", "Manappuram Finance", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("PFC.NS", "Power Finance Corp", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("RECLTD.NS", "REC Limited", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("IDFCFIRSTB.NS", "IDFC First Bank", AssetCategory.INDIAN_STOCKS, "NSE"),
    # ── Telecom / Media ──
    AssetInfo("IDEA.NS", "Vodafone Idea", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("ZEEL.NS", "Zee Entertainment", AssetCategory.INDIAN_STOCKS, "NSE"),
    # ── Others ──
    AssetInfo("DLF.NS", "DLF", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("GODREJPROP.NS", "Godrej Properties", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("OBEROIRLTY.NS", "Oberoi Realty", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("PAGEIND.NS", "Page Industries", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("TRENT.NS", "Trent", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("POLYCAB.NS", "Polycab India", AssetCategory.INDIAN_STOCKS, "NSE"),
    AssetInfo("PIIND.NS", "PI Industries", AssetCategory.INDIAN_STOCKS, "NSE"),
]

# Forex Major Pairs (yfinance symbols)
FOREX_MAJOR: List[AssetInfo] = [
    AssetInfo("EURUSD=X", "EUR/USD", AssetCategory.FOREX_MAJOR, "Forex"),
    AssetInfo("GBPUSD=X", "GBP/USD", AssetCategory.FOREX_MAJOR, "Forex"),
    AssetInfo("USDJPY=X", "USD/JPY", AssetCategory.FOREX_MAJOR, "Forex"),
    AssetInfo("AUDUSD=X", "AUD/USD", AssetCategory.FOREX_MAJOR, "Forex"),
    AssetInfo("USDCHF=X", "USD/CHF", AssetCategory.FOREX_MAJOR, "Forex"),
    AssetInfo("USDCAD=X", "USD/CAD", AssetCategory.FOREX_MAJOR, "Forex"),
    AssetInfo("NZDUSD=X", "NZD/USD", AssetCategory.FOREX_MAJOR, "Forex"),
]

# Forex Minor / Cross Pairs
FOREX_MINOR: List[AssetInfo] = [
    AssetInfo("EURGBP=X", "EUR/GBP", AssetCategory.FOREX_MINOR, "Forex"),
    AssetInfo("EURAUD=X", "EUR/AUD", AssetCategory.FOREX_MINOR, "Forex"),
    AssetInfo("EURJPY=X", "EUR/JPY", AssetCategory.FOREX_MINOR, "Forex"),
    AssetInfo("EURCHF=X", "EUR/CHF", AssetCategory.FOREX_MINOR, "Forex"),
    AssetInfo("GBPJPY=X", "GBP/JPY", AssetCategory.FOREX_MINOR, "Forex"),
    AssetInfo("GBPAUD=X", "GBP/AUD", AssetCategory.FOREX_MINOR, "Forex"),
    AssetInfo("GBPCHF=X", "GBP/CHF", AssetCategory.FOREX_MINOR, "Forex"),
    AssetInfo("AUDJPY=X", "AUD/JPY", AssetCategory.FOREX_MINOR, "Forex"),
    AssetInfo("AUDNZD=X", "AUD/NZD", AssetCategory.FOREX_MINOR, "Forex"),
    AssetInfo("NZDJPY=X", "NZD/JPY", AssetCategory.FOREX_MINOR, "Forex"),
    AssetInfo("CADJPY=X", "CAD/JPY", AssetCategory.FOREX_MINOR, "Forex"),
    AssetInfo("CHFJPY=X", "CHF/JPY", AssetCategory.FOREX_MINOR, "Forex"),
]

# INR Forex Pairs
FOREX_INR: List[AssetInfo] = [
    AssetInfo("INR=X", "USD/INR", AssetCategory.FOREX_INR, "Forex"),
    AssetInfo("EURINR=X", "EUR/INR", AssetCategory.FOREX_INR, "Forex"),
    AssetInfo("GBPINR=X", "GBP/INR", AssetCategory.FOREX_INR, "Forex"),
    AssetInfo("JPYINR=X", "JPY/INR", AssetCategory.FOREX_INR, "Forex"),
]

# Indices
INDICES: List[AssetInfo] = [
    AssetInfo("^NSEI", "NIFTY 50", AssetCategory.INDICES, "NSE"),
    AssetInfo("^NSEBANK", "Bank NIFTY", AssetCategory.INDICES, "NSE"),
    AssetInfo("^BSESN", "BSE SENSEX", AssetCategory.INDICES, "BSE"),
    AssetInfo("^NSEMDCP50", "NIFTY Midcap 50", AssetCategory.INDICES, "NSE"),
    AssetInfo("^CNXIT", "NIFTY IT", AssetCategory.INDICES, "NSE"),
    AssetInfo("^CNXPHARMA", "NIFTY Pharma", AssetCategory.INDICES, "NSE"),
]

# Commodities (yfinance futures symbols)
COMMODITIES: List[AssetInfo] = [
    AssetInfo("GC=F", "Gold Futures", AssetCategory.COMMODITIES, "COMEX"),
    AssetInfo("SI=F", "Silver Futures", AssetCategory.COMMODITIES, "COMEX"),
    AssetInfo("CL=F", "Crude Oil WTI", AssetCategory.COMMODITIES, "NYMEX"),
    AssetInfo("BZ=F", "Brent Crude", AssetCategory.COMMODITIES, "ICE"),
    AssetInfo("NG=F", "Natural Gas", AssetCategory.COMMODITIES, "NYMEX"),
    AssetInfo("HG=F", "Copper Futures", AssetCategory.COMMODITIES, "COMEX"),
    AssetInfo("PL=F", "Platinum Futures", AssetCategory.COMMODITIES, "NYMEX"),
    AssetInfo("ZC=F", "Corn Futures", AssetCategory.COMMODITIES, "CBOT"),
    AssetInfo("ZW=F", "Wheat Futures", AssetCategory.COMMODITIES, "CBOT"),
    AssetInfo("CT=F", "Cotton Futures", AssetCategory.COMMODITIES, "ICE"),
]

# Crypto
CRYPTO: List[AssetInfo] = [
    AssetInfo("BTC-USD", "Bitcoin", AssetCategory.CRYPTO, "Crypto"),
    AssetInfo("ETH-USD", "Ethereum", AssetCategory.CRYPTO, "Crypto"),
    AssetInfo("BNB-USD", "Binance Coin", AssetCategory.CRYPTO, "Crypto"),
    AssetInfo("XRP-USD", "Ripple", AssetCategory.CRYPTO, "Crypto"),
    AssetInfo("SOL-USD", "Solana", AssetCategory.CRYPTO, "Crypto"),
    AssetInfo("ADA-USD", "Cardano", AssetCategory.CRYPTO, "Crypto"),
    AssetInfo("DOGE-USD", "Dogecoin", AssetCategory.CRYPTO, "Crypto"),
    AssetInfo("DOT-USD", "Polkadot", AssetCategory.CRYPTO, "Crypto"),
    AssetInfo("MATIC-USD", "Polygon", AssetCategory.CRYPTO, "Crypto"),
    AssetInfo("AVAX-USD", "Avalanche", AssetCategory.CRYPTO, "Crypto"),
]


# ── Master catalogue (ordered) ──────────────────────────────────────

_ALL_ASSETS: List[AssetInfo] = (
    INDIAN_STOCKS
    + FOREX_MAJOR
    + FOREX_MINOR
    + FOREX_INR
    + INDICES
    + COMMODITIES
    + CRYPTO
)

_CATEGORY_MAP: Dict[AssetCategory, List[AssetInfo]] = {
    AssetCategory.INDIAN_STOCKS: INDIAN_STOCKS,
    AssetCategory.FOREX_MAJOR: FOREX_MAJOR,
    AssetCategory.FOREX_MINOR: FOREX_MINOR,
    AssetCategory.FOREX_INR: FOREX_INR,
    AssetCategory.INDICES: INDICES,
    AssetCategory.COMMODITIES: COMMODITIES,
    AssetCategory.CRYPTO: CRYPTO,
}


# ── Asset Registry ───────────────────────────────────────────────────


class AssetRegistry:
    """Dynamic read-only registry for all tradeable assets.

    The registry is loaded once from the module-level catalogue and provides
    search, category-filter, and lookup helpers used by the dashboard and
    API layers.

    Example::

        registry = AssetRegistry()
        results = registry.search("reliance")
        indian = registry.get_by_category(AssetCategory.INDIAN_STOCKS)
    """

    def __init__(self) -> None:
        """Initialises the registry from the static catalogue."""
        self._assets: List[AssetInfo] = list(_ALL_ASSETS)
        self._symbol_index: Dict[str, AssetInfo] = {
            a.symbol: a for a in self._assets
        }
        logger.info(
            "AssetRegistry initialised with %d assets across %d categories.",
            len(self._assets),
            len(_CATEGORY_MAP),
        )

    # ── Query helpers ────────────────────────────────────────────────

    def get_all(self) -> List[AssetInfo]:
        """Returns every registered asset.

        Returns:
            List of all ``AssetInfo`` objects.
        """
        return list(self._assets)

    def get_by_category(self, category: AssetCategory) -> List[AssetInfo]:
        """Returns all assets belonging to *category*.

        Args:
            category: The ``AssetCategory`` to filter by.

        Returns:
            List of matching ``AssetInfo`` objects.
        """
        return [a for a in self._assets if a.category == category]

    def get_by_symbol(self, symbol: str) -> Optional[AssetInfo]:
        """Exact-match lookup by ticker symbol.

        Args:
            symbol: Provider-specific ticker (case-sensitive).

        Returns:
            ``AssetInfo`` if found, else ``None``.
        """
        return self._symbol_index.get(symbol)

    def search(self, query: str) -> List[AssetInfo]:
        """Case-insensitive prefix / substring search.

        Matches against both ``symbol`` and ``name`` fields.

        Args:
            query: Search string (e.g. ``"reli"`` or ``"EUR"``).

        Returns:
            List of matching ``AssetInfo`` objects.
        """
        if not query:
            return []
        q = query.upper()
        return [
            a
            for a in self._assets
            if q in a.symbol.upper() or q in a.name.upper()
        ]

    def get_symbols_for_category(self, category: AssetCategory) -> List[str]:
        """Returns just the ticker symbols for a category.

        Args:
            category: The ``AssetCategory`` to filter by.

        Returns:
            List of ticker strings.
        """
        return [a.symbol for a in self.get_by_category(category)]

    def get_display_map(
        self, category: Optional[AssetCategory] = None
    ) -> Dict[str, str]:
        """Returns ``{symbol: display_name}`` mapping.

        Args:
            category: Optional filter.  ``None`` returns all assets.

        Returns:
            Dictionary keyed by symbol with human-readable display names.
        """
        assets = self.get_by_category(category) if category else self._assets
        return {a.symbol: a.name for a in assets}

    @staticmethod
    def categories() -> List[AssetCategory]:
        """Returns all available asset categories.

        Returns:
            Ordered list of ``AssetCategory`` values.
        """
        return list(AssetCategory)

    def __len__(self) -> int:
        return len(self._assets)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._symbol_index
