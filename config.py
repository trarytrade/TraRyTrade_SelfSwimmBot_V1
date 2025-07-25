"""Basic configuration for Binance credentials.

The API key and secret can be supplied through environment variables
`BINANCE_API_KEY` and `BINANCE_API_SECRET`. If the variables are not set,
the default placeholder values are used so the rest of the application can
still load without errors.
"""

import os

api_key_binance = os.environ.get("BINANCE_API_KEY", "your api_key_binance")
api_secret_binance = os.environ.get("BINANCE_API_SECRET", "your_api_secret_binance")

