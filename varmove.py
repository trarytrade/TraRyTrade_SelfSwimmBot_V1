#!/usr/bin/env python
"""Trading parameters used by the main bot.

The values can be overridden via the environment variables `TRADE_COIN`,
`TRADE_AMOUNT`, and `TRADE_X`. This makes it easier to adjust settings in
different deployment environments without editing this file directly.
"""

import os

Coin = os.environ.get("TRADE_COIN", "LUMIAUSDT")

def _parse_float(env_value: str, default: float) -> float:
    try:
        return float(env_value)
    except (TypeError, ValueError):
        return default

TradeAmount = _parse_float(os.environ.get("TRADE_AMOUNT"), 24)
TradeX = _parse_float(os.environ.get("TRADE_X"), 6)

