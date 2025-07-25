"""Configuration for TraRyTrade SelfSwimmBot.

API keys can be provided either by editing this file or via environment
variables `BINANCE_API_KEY` and `BINANCE_API_SECRET` for convenience.
"""

import os

# Fallback values are kept for backwards compatibility
api_key_binance = os.getenv("BINANCE_API_KEY", "your api_key_binance")
api_secret_binance = os.getenv("BINANCE_API_SECRET", "your_api_secret_binance")

