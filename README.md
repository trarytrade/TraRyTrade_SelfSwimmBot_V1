# TraRyTrade SelfSwimmBot V1

A single-file trading bot with extended machine learning features. The project is released under the MIT license and aims to provide a starting point for experimenting with automated Binance trading.

## Quick Setup

1. **Clone the repository** and optionally create a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # on Windows use .venv\Scripts\activate
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure API keys** by exporting environment variables or editing `config.py`:
   ```bash
   export BINANCE_API_KEY="YOUR_KEY"
   export BINANCE_API_SECRET="YOUR_SECRET"
   ```
   The bot will read these variables on start up.
4. **Optional configuration**: edit `varmove.py` to set the trading symbol and sizing.
5. **Run the bot**:
   ```bash
   python3 TraRyTrade_SelfSwimm_V1.py
   ```

## What's New

- `safe_sleep()` helper for graceful shutdowns.
- Environment variable support for API keys.
- Example `requirements.txt` for easy installation.

Use at your own risk. Tweak variables and strategy parameters to suit your needs.
