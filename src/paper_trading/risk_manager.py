"""
Risk management and position sizing module for paper trading.

Enforces risk management rules, calculates Stop Loss and Take Profit levels,
and computes AI-driven position sizing based on portfolio equity and risk parameters.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class RiskManager:
    """Enforces risk rules and computes recommended position parameters for paper trades.

    Attributes:
        max_risk_per_trade_pct: Maximum portfolio equity risk percentage per trade (e.g., 2.0%).
        risk_reward_ratio: Target risk-to-reward ratio for Take Profit (e.g., 2.0).
        atr_sl_multiplier: Multiplier applied to ATR for Stop Loss calculation (e.g., 2.0).
    """

    def __init__(
        self,
        max_risk_per_trade_pct: float = 2.0,
        risk_reward_ratio: float = 2.0,
        atr_sl_multiplier: float = 2.0,
    ) -> None:
        """Initialises the RiskManager with custom risk settings.

        Args:
            max_risk_per_trade_pct: Max percentage of portfolio capital to risk on a trade.
            risk_reward_ratio: Ratio of expected gain to risk distance.
            atr_sl_multiplier: ATR multiplier for dynamic Stop Loss distance.
        """
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.risk_reward_ratio = risk_reward_ratio
        self.atr_sl_multiplier = atr_sl_multiplier
        logger.info(
            "RiskManager initialised (max_risk=%.1f%%, RR=%.1f, ATR_mult=%.1f).",
            max_risk_per_trade_pct,
            risk_reward_ratio,
            atr_sl_multiplier,
        )

    def calculate_sl_tp(
        self,
        entry_price: float,
        action: str = "BUY",
        atr: Optional[float] = None,
        default_sl_pct: float = 2.5,
    ) -> Tuple[float, float]:
        """Calculates Stop Loss and Take Profit price levels.

        Args:
            entry_price: Market entry price.
            action: Trading action (BUY or SELL).
            atr: Optional Average True Range value for dynamic SL.
            default_sl_pct: Fallback Stop Loss percentage if ATR is unavailable.

        Returns:
            Tuple of (stop_loss_price, take_profit_price).
        """
        if entry_price <= 0:
            return 0.0, 0.0

        if atr is not None and atr > 0:
            sl_distance = atr * self.atr_sl_multiplier
        else:
            sl_distance = entry_price * (default_sl_pct / 100.0)

        tp_distance = sl_distance * self.risk_reward_ratio

        action_upper = action.upper()
        if action_upper == "SELL":
            stop_loss = round(entry_price + sl_distance, 2)
            take_profit = round(entry_price - tp_distance, 2)
        else:  # BUY or HOLD default
            stop_loss = round(entry_price - sl_distance, 2)
            take_profit = round(entry_price + tp_distance, 2)

        return stop_loss, take_profit

    def calculate_position_size(
        self,
        portfolio_equity: float,
        entry_price: float,
        stop_loss: float,
        confidence: float = 1.0,
        available_cash: Optional[float] = None,
    ) -> int:
        """Calculates recommended position size (shares/units) based on risk budget.

        Risk Amount = Portfolio Equity * (max_risk_per_trade_pct / 100) * confidence
        Quantity = Risk Amount / |entry_price - stop_loss|

        Args:
            portfolio_equity: Current total portfolio equity.
            entry_price: Entry price.
            stop_loss: Stop Loss price level.
            confidence: Model confidence score (0.0 to 1.0) to scale position.
            available_cash: Optional cash constraint.

        Returns:
            Recommended position quantity (integer count >= 1 or 0 if invalid).
        """
        if portfolio_equity <= 0 or entry_price <= 0:
            return 0

        risk_distance = abs(entry_price - stop_loss)
        if risk_distance <= 0:
            risk_distance = entry_price * 0.02  # fallback 2% distance

        # Risk budget scaled by confidence
        scaled_confidence = max(0.5, min(1.0, confidence))
        risk_budget = (
            portfolio_equity * (self.max_risk_per_trade_pct / 100.0) * scaled_confidence
        )

        quantity = int(risk_budget / risk_distance)

        # Cash cap validation if provided
        cash = available_cash if available_cash is not None else portfolio_equity
        max_shares_by_cash = int(cash / entry_price)

        quantity = min(quantity, max_shares_by_cash)
        return max(0, quantity)

    def generate_trade_recommendation(
        self,
        symbol: str,
        entry_price: float,
        action: str,
        confidence: float,
        portfolio_equity: float,
        available_cash: float,
        atr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generates a complete AI trade recommendation payload.

        Args:
            symbol: Ticker symbol.
            entry_price: Current market price.
            action: Trading action (BUY, SELL, HOLD).
            confidence: Model prediction confidence score.
            portfolio_equity: Current total portfolio equity.
            available_cash: Available cash.
            atr: Optional ATR value for technical SL computation.

        Returns:
            Dictionary containing:
                - ``symbol``
                - ``action``
                - ``recommended_size``
                - ``risk_pct``
                - ``stop_loss``
                - ``take_profit``
                - ``confidence_score``
                - ``entry_price``
        """
        stop_loss, take_profit = self.calculate_sl_tp(
            entry_price=entry_price, action=action, atr=atr
        )

        qty = self.calculate_position_size(
            portfolio_equity=portfolio_equity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            confidence=confidence,
            available_cash=available_cash,
        )

        # Actual risk % for this size
        risk_distance = abs(entry_price - stop_loss)
        actual_risk_amount = qty * risk_distance
        actual_risk_pct = (
            round((actual_risk_amount / portfolio_equity) * 100.0, 2)
            if portfolio_equity > 0
            else 0.0
        )

        return {
            "symbol": symbol,
            "action": action,
            "recommended_size": qty,
            "risk_pct": actual_risk_pct,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence_score": round(confidence, 4),
            "entry_price": entry_price,
        }

    def validate_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        available_cash: float,
    ) -> Tuple[bool, str]:
        """Validates if an order complies with risk rules and available cash.

        Args:
            symbol: Ticker symbol.
            action: Order action (BUY/SELL).
            quantity: Requested quantity.
            price: Order execution price.
            available_cash: Current available cash balance.

        Returns:
            Tuple of (is_valid, reason_string).
        """
        if quantity <= 0:
            return False, "Order quantity must be greater than zero."

        if price <= 0:
            return False, "Invalid price."

        if action.upper() == "BUY":
            required_cash = quantity * price
            if required_cash > available_cash:
                return (
                    False,
                    f"Insufficient funds: Required ₹{required_cash:,.2f}, Available ₹{available_cash:,.2f}.",
                )

        return True, "Order validated."
