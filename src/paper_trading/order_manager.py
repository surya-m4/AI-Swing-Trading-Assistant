"""
Order execution and management engine for paper trading.

Handles order placement, execution validation, position closing,
and Stop Loss / Take Profit modifications for Indian Stocks and Forex pairs.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from src.paper_trading.portfolio import Portfolio
from src.paper_trading.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages order execution, order validation, and position closing.

    Attributes:
        portfolio: Portfolio state manager instance.
        risk_manager: Risk manager instance.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        risk_manager: Optional[RiskManager] = None,
    ) -> None:
        """Initialises the OrderManager.

        Args:
            portfolio: Portfolio instance.
            risk_manager: RiskManager instance. Defaults to default RiskManager if None.
        """
        self.portfolio = portfolio
        self.risk_manager = risk_manager if risk_manager is not None else RiskManager()

    def _determine_asset_class(self, symbol: str) -> str:
        """Determines asset category from ticker string convention.

        Args:
            symbol: Ticker symbol (e.g. RELIANCE.NS, EURUSD=X, BTC-USD).

        Returns:
            Asset class string: ``INDIAN_STOCKS``, ``FOREX``, ``CRYPTO``, or ``EQUITY``.
        """
        sym = symbol.upper()
        if sym.endswith(".NS") or sym.startswith("^"):
            return "INDIAN_STOCKS"
        if "=X" in sym or "/" in sym or sym.endswith("INR"):
            return "FOREX"
        if "-USD" in sym:
            return "CRYPTO"
        return "EQUITY"

    def place_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Places and executes a paper order.

        Args:
            symbol: Asset ticker symbol.
            action: Order action (BUY or SELL).
            quantity: Order quantity.
            price: Execution price.
            stop_loss: Optional Stop Loss price level.
            take_profit: Optional Take Profit price level.

        Returns:
            Order execution payload dict.
        """
        action_upper = action.upper()
        asset_class = self._determine_asset_class(symbol)

        # Validate order if BUY
        if action_upper == "BUY":
            is_valid, reason = self.risk_manager.validate_order(
                symbol=symbol,
                action=action_upper,
                quantity=quantity,
                price=price,
                available_cash=self.portfolio.cash,
            )
            if not is_valid:
                logger.warning("Order validation failed for %s: %s", symbol, reason)
                return {"status": "REJECTED", "reason": reason, "symbol": symbol}

        order_id = f"ORD_{uuid.uuid4().hex[:8].upper()}"

        if action_upper == "BUY":
            cost = quantity * price
            self.portfolio.update_cash(-cost)

            pos = self.portfolio.add_position(
                symbol=symbol,
                asset_class=asset_class,
                quantity=quantity,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_type="LONG",
            )

            self.portfolio.db.record_order(
                order_id=order_id,
                symbol=symbol,
                action="BUY",
                quantity=quantity,
                price=price,
                status="FILLED",
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

            logger.info(
                "BUY order filled: %d %s @ ₹%.2f (Cost: ₹%.2f).",
                quantity,
                symbol,
                price,
                cost,
            )
            return {
                "order_id": order_id,
                "status": "FILLED",
                "action": "BUY",
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "cost": round(cost, 2),
                "position": pos,
            }

        elif action_upper in ("SELL", "CLOSE"):
            return self.close_position(symbol=symbol, exit_price=price)

        return {"status": "REJECTED", "reason": f"Unsupported action '{action}'"}

    def close_position(
        self, symbol: str, exit_price: float
    ) -> Dict[str, Any]:
        """Closes an active open position and records realized P/L.

        Args:
            symbol: Ticker symbol of the open position.
            exit_price: Market exit execution price.

        Returns:
            Dictionary containing closed trade metrics or failure reason.
        """
        pos = self.portfolio.get_position(symbol)
        if not pos:
            return {
                "status": "REJECTED",
                "reason": f"No open position found for {symbol}.",
                "symbol": symbol,
            }

        qty = int(pos.get("quantity", 0))
        entry_price = float(pos.get("entry_price", 0.0))
        opened_at = str(pos.get("created_at", datetime.utcnow().isoformat() + "Z"))

        if pos.get("position_type", "LONG") == "SHORT":
            realized_pnl = (entry_price - exit_price) * qty
        else:
            realized_pnl = (exit_price - entry_price) * qty

        realized_pnl = round(realized_pnl, 2)
        cost_basis = qty * entry_price
        return_pct = (
            round((realized_pnl / cost_basis) * 100.0, 2) if cost_basis > 0 else 0.0
        )

        # Return capital + profit/loss to cash
        proceeds = cost_basis + realized_pnl
        self.portfolio.update_cash(proceeds)

        trade_id = f"TRD_{uuid.uuid4().hex[:8].upper()}"
        self.portfolio.db.record_closed_trade(
            trade_id=trade_id,
            symbol=symbol,
            action="CLOSE",
            quantity=qty,
            entry_price=entry_price,
            exit_price=exit_price,
            realized_pnl=realized_pnl,
            return_pct=return_pct,
            opened_at=opened_at,
        )

        self.portfolio.remove_position(symbol)

        logger.info(
            "Closed position for %s: Qty=%d, Entry=%.2f, Exit=%.2f, PnL=₹%.2f (%.2f%%).",
            symbol,
            qty,
            entry_price,
            exit_price,
            realized_pnl,
            return_pct,
        )

        return {
            "status": "CLOSED",
            "trade_id": trade_id,
            "symbol": symbol,
            "quantity": qty,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "realized_pnl": realized_pnl,
            "return_pct": return_pct,
            "proceeds": round(proceeds, 2),
        }

    def modify_sl_tp(
        self,
        symbol: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Modifies Stop Loss and Take Profit levels for an open position.

        Args:
            symbol: Ticker symbol.
            stop_loss: New Stop Loss level.
            take_profit: New Take Profit level.

        Returns:
            Dictionary with status and updated position.
        """
        success = self.portfolio.modify_position_sl_tp(
            symbol=symbol, stop_loss=stop_loss, take_profit=take_profit
        )

        if not success:
            return {
                "status": "REJECTED",
                "reason": f"No open position found for {symbol}.",
            }

        pos = self.portfolio.get_position(symbol)
        logger.info(
            "Modified SL/TP for %s: SL=%.2f, TP=%.2f.",
            symbol,
            stop_loss or 0.0,
            take_profit or 0.0,
        )
        return {
            "status": "SUCCESS",
            "symbol": symbol,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position": pos,
        }
