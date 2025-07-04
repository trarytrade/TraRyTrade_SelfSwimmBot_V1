#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TraRyTrade — Public Open-Source Release
--------------------------------------------------------------------------------
⚠️  **ATTENTION**  
   - This bot may require further fine-tuning in code and `setable Vars`.  
   - Out of the box it may not be profitable—use at your own risk!  

Copyright:  © 2025 TraRyTrade  
License:    MIT (see LICENSE file)  
Version:    1.0.0  
Date:       2025-07-04  

Features:
  - Fully autonomous trading engine: continuously buys, sells, scales, and grid-trades without manual intervention
  - Advanced ML signals including:
      • MOVE_STOP_BREAKEVEN — auto-move stop loss to breakeven when profitable
      • ADJUST_MAX_POS_n — dynamically resize maximum position limits on the fly
      • TAKE_HALF / TAKE_QUARTER — partial exits for precise risk control
  - Multi-lot order support with flexible position sizing (open 1–3 lots, scale-in, flip)
  - Integrated 14-period RSI + TSI (True Strength Index) indicators for momentum filtering
  - Real-time “wave score” volatility measure + regime detection for adaptive behavior
  - Built-in grid-trading mode to layer limit orders across price levels
  - Robust async architecture:
      • Live Binance WebSocket streams for trades & liquidations
      • Non-blocking order execution in subprocesses
      • Periodic model retraining and JSON performance dumps
  - Detailed logging & performance alerts:
      • “Profitable trade” messages when unrealized PnL exceeds threshold
      • Structured ML-decision and sync logs for auditability
  - Self-healing data management:
      • Atomic JSON writes, file-locking for IPC, automatic history backfill
      • Graceful handling of API errors and reconnections
  - Zero-warranty “AS IS” release—use at your own risk
"""




import aiohttp, asyncio, json, time, traceback, os, sys, math, datetime, importlib, random, subprocess, signal, tempfile
from collections import deque
from decimal import Decimal
from typing import List, Optional, Deque, Dict, Tuple

# Third‑party modules
import requests, hashlib, numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError
from binance.client import Client

# External configuration
import config, varmove


import fcntl
###############################################################################
# GLOBAL SETTINGS & CONSTANTS
###############################################################################
REAL_TRADING              = True
API_KEY                   = config.api_key_binance
API_SECRET                = config.api_secret_binance
LEVERAGE                  = 50

API_error_ON = 0


SYMBOL                    = varmove.Coin.upper()
TRADE_UNIT_SIZE           = float(varmove.TradeAmount)

MAX_POSITION_UNITS_ABS    = float(varmove.TradeX)    # Hard limit – never exceed TradeX
MAX_POSITION_UNITS_CURRENT= float(varmove.TradeX)    # Current dynamic maximum position

STOP_LOSS_ML              = 0.0042     # e.g., forced stop if drawdown reaches ~1%
MOVE_STOP_BREAKEVEN_level = 0.0072
move_stop_to_breakevenValveLONG = 1.003
move_stop_to_breakevenValveSHORT = 0.997


MIN_DATA_FOR_TRAIN        = 2




RETRAIN_INTERVAL_SEC      = 300
periodic_check_modell_swicth_time = 999999
MIN_COOLDOWN_SEC_MAIN     = 0.1
MIN_COOLDOWN_SEC_SUB      = 0.1



PERF_WRITE_INTERVAL       = 120         # seconds between performance file writes
PROFIT_THRESHOLD          = 0.03       # log if unrealized profit is above 5%

LOG_FILE                  = "TraRyTrade_SelfSwimm_V1_superrefined.log"
POSITION_FILE             = "position.json"
TRADES_FILE               = "TraRyTrade_SelfSwimm_V1_trades.json"
MODEL_FILE                = "TraRyTrade_SelfSwimm_V1_model.pkl"
SCALER_FILE               = "TraRyTrade_SelfSwimm_V1_scaler.pkl"
TRADE_DATA_LOG            = "trade_data.log"   # shared trade log file for IPC

LOG_POSITION_SYNC         = "POSSync.log"
ML_DECISION_LOG           = "MLDecision.log"
BTS_POS_FILE              = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BTSposLive.json')

# Extended ML actions
ACTIONS = [
    "HOLD",
    "OPEN_LONG_1", "OPEN_LONG_2", "OPEN_LONG_3",
    "OPEN_SHORT_1", "OPEN_SHORT_2", "OPEN_SHORT_3",
    "SCALE_IN_LONG", "SCALE_IN_SHORT",
    "PARTIAL_CLOSE", "CLOSE_FULL",
    "FLIP_LONG", "FLIP_SHORT",
    "MOVE_STOP_BREAKEVEN",
    "ADJUST_MAX_POS_1", "ADJUST_MAX_POS_2", "ADJUST_MAX_POS_3",
    "ADJUST_MAX_POS_4", "ADJUST_MAX_POS_5",
    "TAKE_HALF", "TAKE_QUARTER"
]


###############################################################################
# TECHNICAL INDICATOR: RSI Calculation
###############################################################################
def compute_RSI(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the 14‑period RSI (default) and fill missing with 50 for very short series.
    """
    delta = prices.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.reindex(prices.index).fillna(50)



###############################################################################
# ASYNC LOGGING FOR ML DECISIONS
###############################################################################
async def async_log_ml_decision(action, price, prev_units, new_units, executed):
    """
    Logs the ML decision into MLDecision.log asynchronously in a structured format.
    """
    await asyncio.sleep(0.25)
    data = json.dumps([action, price, prev_units, new_units, executed])
    code_snippet = f"""
import json, datetime, os

def log_ml_decision(action, price, prev_units, new_units, executed):
    if not os.path.exists("MLDecision.log"):
        with open("MLDecision.log", "w") as f:
            f.write("==== ML Decision Log ====\\n")
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    status = "SUCCESS" if executed else "FAILED"
    log_entry = (
        "================= ML DECISION LOG =================\\n"
        f"Timestamp           : {{ts}}\\n"
        "--------------------------------------------------\\n"
        f"ML Signal Given     : {{action}}\\n"
        f"Execution Status    : {{status}}\\n"
        "--------------------------------------------------\\n"
        f"Previous Pos Units  : {{prev_units:.2f}}\\n"
        f"New Pos Units       : {{new_units:.2f}}\\n"
        f"Price at Decision   : {{price:.6f}}\\n"
        "==================================================\\n"
    )
    with open("MLDecision.log", "a") as f:
        f.write(log_entry + "\\n")

args = json.loads('{data}')
log_ml_decision(*args)
"""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", code_snippet,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    if stderr:
        print(f"[MLDecision Error]: {stderr.decode().strip()}")
    if stdout:
        print(f"[MLDecision Output]: {stdout.decode().strip()}")

###############################################################################
# LOGGER
###############################################################################
class Logger:
    def __init__(self, path=LOG_FILE):
        self.path = path
        self.buffer = []
        self.last_flush = time.time()

    def log(self, msg: str, console=True):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} | {msg}"
        if console:
            print(line)
        self.buffer.append(line)

    def flush(self, force=False):
        now = time.time()
        if not force and (now - self.last_flush < 5.0):
            return
        self.last_flush = now
        if not self.buffer:
            return
        text = "\n".join(self.buffer) + "\n"
        self.buffer.clear()
        try:
            if os.path.exists(self.path) and os.path.getsize(self.path) > 1_000_000:
                os.remove(self.path)
            with open(self.path, "a") as f:
                f.write(text)
        except Exception as e:
            print(f"[Logger] Error writing log: {e}")

def load_positions_registry(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return {}

###############################################################################
# API FLAG CHECK IF TRUE ALL IS OK   if   "Error": false
###############################################################################


def check_api_error():
    """
    Reads ApiState.json (expected to contain {"Error": <bool>}), 
    defaults to no error (False), and returns:
      0 if no error (Error == False),
      1 if Error == True or on any read/parse failure.
    """
    # 1) Default to no error
    error_flag = False

    # 2) Locate the JSON file two levels up
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "..", "..", "ApiState.json")

    # 3) Try to read & parse
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        # Treat any I/O or parsing failure as an error
        print(f"Error accessing or parsing ApiState.json ({e}), marking error_flag = True")
        error_flag = True
    else:
        # 4) Flip to True only if the JSON explicitly says Error: true
        if data.get("Error") is True:
            error_flag = True

    # 5) Return 1 on error, 0 otherwise
    return 1 if error_flag else 0


###############################################################################
# Fetch DB results Real trades results
###############################################################################


import sqlite3
import os
from typing import List, Dict
# Construct the DB path dynamically
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, os.pardir, os.pardir))
DB_PATH = os.path.join(_project_root, "instance", "users.db")





###############################################################################
# LONG TERM MARKET HIST
###############################################################################


import os
import time
import json
import datetime
import pandas as pd
from binance.client import Client


# LONG TERM MARKET HIST

import os, time, json, datetime
import pandas as pd
from binance.client import Client

# Configuration
HIST_DAYS_BACK = 90         # number of days of history to keep
HIST_RES_MIN   = 5         # bar resolution in minutes
HIST_FILE      = "long_hist_trades.json"

class LongHistoryManager:
    """
    Manages a rolling window of historical 15-minute bars for a given symbol,
    stored in JSON. Automatically backfills missing bars on startup and
    persists updates on each new trade.
    """

    def __init__(self, client: Client, symbol: str, logger=None):
        self.client = client
        self.symbol = symbol
        self.logger = logger or Logger()        # ← store a logger
        self._last_persisted_period = None

        # Load existing history or fetch initial window
        if os.path.exists(HIST_FILE):
            with open(HIST_FILE, "r") as f:
                bars = json.load(f)
            self.df = pd.DataFrame(bars)
            self.df['open_time'] = pd.to_datetime(self.df['open_time'], utc=True)
        else:
            self._fetch_initial()

        # Prune to last HIST_DAYS_BACK days
        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=HIST_DAYS_BACK)
        self.df = self.df[self.df.open_time >= cutoff].reset_index(drop=True)

        # Backfill any missing intervals since last stored bar
        if not self.df.empty:
            last = self.df.open_time.max()
            if pd.Timestamp.now(tz='UTC') - last >= pd.Timedelta(minutes=HIST_RES_MIN):
                self._fetch_missing(start_after=last)

        # Persist the up-to-date history
        self._persist()

    def get_df(self):
        """Expose the current DataFrame for external use."""
        return self.df

    def _fetch_initial(self):
        """
        Fetch the past HIST_DAYS_BACK days of 15m bars by paging through
        the Binance API (max 1500-bars per call).
        """
        end_ms   = int(time.time() * 1000)
        start_ms = int((time.time() - 86400 * HIST_DAYS_BACK) * 1000)
        all_bars = []
        MAX_LIMIT = 1500

        while True:
            chunk = self.client.futures_klines(
                symbol=self.symbol,
                interval=f"{HIST_RES_MIN}m",
                startTime=start_ms,
                endTime=end_ms,
                limit=MAX_LIMIT
            )
            if not chunk:
                break

            for r in chunk:
                ts = datetime.datetime.utcfromtimestamp(r[0]/1000).replace(tzinfo=datetime.timezone.utc)
                all_bars.append({
                    'open_time': ts,
                    'open':      float(r[1]),
                    'high':      float(r[2]),
                    'low':       float(r[3]),
                    'close':     float(r[4]),
                    'volume':    float(r[5])
                })

            if len(chunk) < MAX_LIMIT:
                break

            start_ms = chunk[-1][0] + HIST_RES_MIN * 60 * 1000
            time.sleep(0.1)

        self.df = pd.DataFrame(all_bars)
        self.df['open_time'] = pd.to_datetime(self.df['open_time'], utc=True)
        self._persist()

    def _fetch_missing(self, start_after: pd.Timestamp):
        """Fetch any bars missing since the given timestamp and append to df."""
        start = int((start_after + pd.Timedelta(minutes=HIST_RES_MIN)).timestamp() * 1000)
        end = int(time.time() * 1000)
        raw = self.client.futures_klines(
            symbol=self.symbol,
            interval=f"{HIST_RES_MIN}m",
            startTime=start,
            endTime=end
        )
        new = []
        for r in raw:
            ts = datetime.datetime.utcfromtimestamp(r[0]/1000).replace(tzinfo=datetime.timezone.utc)
            new.append({
                'open_time': ts,
                'open':      float(r[1]),
                'high':      float(r[2]),
                'low':       float(r[3]),
                'close':     float(r[4]),
                'volume':    float(r[5])
            })
        if new:
            df_new = pd.DataFrame(new)
            df_new['open_time'] = pd.to_datetime(df_new['open_time'], utc=True)
            self.df = pd.concat([self.df, df_new], ignore_index=True)
            self.df = self.df.drop_duplicates('open_time').sort_values('open_time').reset_index(drop=True)

    def append_trade(self, price: float, qty: float, ts: float):
        """Aggregate a single trade into the current 15-minute bar and persist."""
        t = pd.Timestamp(ts, unit='s', tz='UTC')
        period_start = t.floor(f"{HIST_RES_MIN}T")

        if self.df.empty or self.df.iloc[-1].open_time != period_start:
            new = {
                'open_time': period_start,
                'open':      price,
                'high':      price,
                'low':       price,
                'close':     price,
                'volume':    qty
            }
            self.df = pd.concat([self.df, pd.DataFrame([new])], ignore_index=True)
        else:
            i = self.df.index[-1]
            self.df.at[i, 'high']   = max(self.df.at[i, 'high'], price)
            self.df.at[i, 'low']    = min(self.df.at[i, 'low'],  price)
            self.df.at[i, 'close']  = price
            self.df.at[i, 'volume'] += qty

        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=HIST_DAYS_BACK)
        self.df = self.df[self.df.open_time >= cutoff].reset_index(drop=True)

        # only persist on new bucket
        if period_start != self._last_persisted_period:
            self._persist()
            self._last_persisted_period = period_start

    def _persist(self):
        """Write the current DataFrame to the JSON file with ISO timestamps."""
        records = self.df.copy()
        records['open_time'] = records['open_time'].dt.tz_convert('UTC').apply(lambda ts: ts.isoformat())
        temp = HIST_FILE + ".tmp"

        try:
            with open(temp, 'w') as f:
                json.dump(records.to_dict(orient='records'), f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # only replace if temp actually got created
            if os.path.exists(temp):
                os.replace(temp, HIST_FILE)
            else:
                self.logger.log(f"[LongHistoryManager] Temp file {temp} not found; skipping replace.")

        except Exception as e:
            self.logger.log(f"[LongHistoryManager._persist] Error writing {temp} → {HIST_FILE}: {e}")




###############################################################################
# ASYNC TRADING ORDER MANAGER
###############################################################################
class TradingBotAsyncManager:
    def __init__(self):
        self.session = None
        self.API_error_ON  = 0




    async def init_session(self):
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    def fire_and_forget_order(self, url: str, headers: dict, body: dict):
        """
        Fire an HTTP POST in a subprocess to avoid blocking the main event loop.
        """
        asyncio.create_task(self._launch_in_subprocess(url, headers, body))

    async def _launch_in_subprocess(self, url: str, headers: dict, body: dict):
        try:
            body_str = json.dumps(body)
            command = [
                'python3', '-c', f"""
import json, aiohttp, asyncio

async def post_request(url, headers, body):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, headers=headers, data=body) as resp:
                response_text = await resp.text()
                print(f"Order status={{resp.status}}, response={{response_text}}")
        except Exception as e:
            print(f"Error sending order: {{e}}")

asyncio.run(post_request('{url}', {json.dumps(headers)}, '{body_str}'))
"""
            ]
            subprocess.Popen(command)
        except Exception as e:
            print(f"[TradingBot] Subprocess launch error: {e}")




###############################################################################
# FILE LOCK TRADE APPEND
###############################################################################




def append_trade_line(file_path: str, trade: dict):
    """
    Append a JSON-encoded trade as one line to file_path using an exclusive lock.
    """
    line = json.dumps(trade) + "\n"
    try:
        with open(file_path, "a") as f:
            # Acquire an exclusive lock
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()      # push into OS buffers
                os.fsync(f.fileno())  # ensure it's on disk
            finally:
                # Always release the lock, even if write/fsync fail
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")


###############################################################################
# POSITION MANAGER
###############################################################################
# Updated PositionManager with full compatibility and precise PnL/fee tracking

class PositionManager:
    def __init__(
        self,
        maker_fee: float = 0.04,
        trade_unit_size: float = 1.0,
        stop_loss_pct: float = STOP_LOSS_ML,  
        logger=None                          
    ):
        self.logger = logger
        
        # Core state
        self.position_units       = 0.0
        self.avg_entry_price      = 0.0
        self.realized_pnl         = 0.0
        self.open_fee_accum       = 0.0
        
        # Legacy run-PnL fields
        self.buy_run_pnl          = 0.0
        self.sell_run_pnl         = 0.0
        
        # Position metadata
        self.bars_in_position     = 0
        self.stop_moved_to_breakeven = False
        self.stop_loss            = 0.0
        self.side                 = "FLAT"
        
        # Fees & sizing
        self.maker_fee            = maker_fee
        self.trade_unit_size      = trade_unit_size
        self.stop_loss_pct        = stop_loss_pct

    # ── Legacy API ──
    def reset_run_pnl(self):
        if self.is_long():
            self.buy_run_pnl = 0.0
        elif self.is_short():
            self.sell_run_pnl = 0.0

    def update_side_from_units(self):
        if self.position_units > 0:
            self.side = "BUYRUN"
            self.sell_run_pnl = 0.0
            # Reflect actual run-PnL
            self.buy_run_pnl = self.get_unrealized_pnl(self.avg_entry_price + 0)
        elif self.position_units < 0:
            self.side = "SELLRUN"
            self.buy_run_pnl = 0.0
            self.sell_run_pnl = self.get_unrealized_pnl(self.avg_entry_price + 0)
        else:
            self.side = "FLAT"
            self.buy_run_pnl = 0.0
            self.sell_run_pnl = 0.0

    def is_long(self) -> bool:
        return self.position_units > 0

    def is_short(self) -> bool:
        return self.position_units < 0

    def is_flat(self) -> bool:
        return abs(self.position_units) < 1e-8

    def increment_bars_in_position(self):
        if not self.is_flat():
            self.bars_in_position += 1
        else:
            self.bars_in_position = 0

    # ── Core helpers ──
    def _compute_fee(self, units: float, price: float) -> float:
        return abs(units) * price * self.trade_unit_size * self.maker_fee

    # ── Adjust position with precise PnL & fee handling ──
    def adjust_position(self, delta_units: float, fill_price: float):
        old = self.position_units
        new = old + delta_units
        fee = self._compute_fee(delta_units, fill_price)

        # Fully close
        if abs(new) < 1e-8:
            gross = (fill_price - self.avg_entry_price) * abs(old) * self.trade_unit_size
            pnl_closed = gross if old > 0 else -gross
            self.realized_pnl += pnl_closed
            self.realized_pnl -= self.open_fee_accum
            # reset everything
            self.position_units       = 0.0
            self.avg_entry_price      = 0.0
            self.open_fee_accum       = 0.0
            self.stop_loss            = 0.0
            self.bars_in_position     = 0
            self.stop_moved_to_breakeven = False
            self.update_side_from_units()
            return

        # Flip side
        if old != 0 and (old * new) < 0:
            gross = (fill_price - self.avg_entry_price) * abs(old) * self.trade_unit_size
            pnl_closed = gross if old > 0 else -gross
            self.realized_pnl += pnl_closed
            self.realized_pnl -= self.open_fee_accum
            # reset and recurse
            self.open_fee_accum       = 0.0
            self.bars_in_position     = 0
            self.position_units       = 0.0
            self.avg_entry_price      = 0.0
            self.stop_loss            = 0.0
            self.stop_moved_to_breakeven = False
            self.update_side_from_units()
            return self.adjust_position(new, fill_price)

        # Partial close (same side)
        if abs(new) < abs(old):
            closed = abs(old) - abs(new)
            gross = (fill_price - self.avg_entry_price) * closed * self.trade_unit_size
            pnl_closed = gross if old > 0 else -gross
            self.realized_pnl += pnl_closed
            # prorate open fees
            ratio = abs(new) / abs(old)
            self.open_fee_accum *= ratio
            self.position_units = new
            self.update_side_from_units()
            return

        # Add units (same side)
        added = abs(new) - abs(old)
        total = abs(old) + added
        self.avg_entry_price = (
            self.avg_entry_price * abs(old)
            + fill_price * added
        ) / total
        self.open_fee_accum += fee
        self.position_units = new
        # update stop loss
        if self.is_long():
            self.stop_loss = self.avg_entry_price * (1 - self.stop_loss_pct)
        else:
            self.stop_loss = self.avg_entry_price * (1 + self.stop_loss_pct)

        self.update_side_from_units()

    def move_stop_to_breakeven(self):
        if self.is_flat() or self.avg_entry_price <= 0:
            return
        if self.is_long():
            self.stop_loss = self.avg_entry_price* move_stop_to_breakevenValveLONG  
        else:
            self.stop_loss = self.avg_entry_price* move_stop_to_breakevenValveSHORT
        self.logger.log(f"[PosMgr]  avg_entry_price  {self.avg_entry_price}   ---- Moved stop to breakeven: new_stop_loss={self.stop_loss:.6f}")
        self.stop_moved_to_breakeven = True

    # ── PnL readers ──
    def get_unrealized_pnl(self, mark_price: float) -> float:
        if self.is_flat():
            return 0.0
        gross = (mark_price - self.avg_entry_price) * abs(self.position_units) * self.trade_unit_size
        gross = gross if self.is_long() else -gross
        return gross - self.open_fee_accum

    def get_unrealized_pct(self, mark_price: float) -> float:
        if self.is_flat() or self.avg_entry_price == 0:
            return 0.0
        return ((mark_price - self.avg_entry_price) / self.avg_entry_price) if self.is_long() \
               else ((self.avg_entry_price - mark_price) / self.avg_entry_price)

    # ── Persistence ──
    def to_dict(self) -> dict:
        return {
            "position_units": self.position_units,
            "avg_entry_price": self.avg_entry_price,
            "realized_pnl": self.realized_pnl,
            "open_fee_accum": self.open_fee_accum,
            "buy_run_pnl": self.buy_run_pnl,
            "sell_run_pnl": self.sell_run_pnl,
            "bars_in_position": self.bars_in_position,
            "stop_moved_to_breakeven": self.stop_moved_to_breakeven,
            "stop_loss": self.stop_loss,
            "side": self.side,
        }

    def from_dict(self, d: dict):
        self.position_units       = d.get("position_units", 0.0)
        self.avg_entry_price      = d.get("avg_entry_price", 0.0)
        self.realized_pnl         = d.get("realized_pnl", 0.0)
        self.open_fee_accum       = d.get("open_fee_accum", 0.0)
        self.buy_run_pnl          = d.get("buy_run_pnl", 0.0)
        self.sell_run_pnl         = d.get("sell_run_pnl", 0.0)
        self.bars_in_position     = d.get("bars_in_position", 0)
        self.stop_moved_to_breakeven = d.get("stop_moved_to_breakeven", False)
        self.stop_loss            = d.get("stop_loss", 0.0)
        self.side                 = d.get("side", "FLAT")
        self.update_side_from_units()





###############################################################################
# ADVANCED WAVE COMPUTER
###############################################################################
class AdvancedWaveComputer:
    def __init__(self, maxlen=200):
        self.maxlen = maxlen
        self.buffer: Deque[Tuple[float, float]] = deque(maxlen=maxlen)

    def add_trade(self, px: float, qty: float):
        self.buffer.append((px, qty))

    def compute_wave_score(self) -> float:
        """
        Simple measure of volatility by price standard deviation.
        """
        if len(self.buffer) < 5:
            return 0.0
        arr = list(self.buffer)
        return float(np.std([p for (p, _) in arr]))

###############################################################################
# LABELING FUNCTIONS
###############################################################################
def improved_label_simulation(live_trades: List[dict], scale_in_min_upnl: float = 0.0):
    """
    Simple simulation that reads 'label' in the trades, tries to track position
    changes, and modifies future 'label' entries if certain conditions meet (like
    forced stop, partial exit, etc.) 
    """
    pos_units = 0.0
    avg_px = 0.0
    bars_in_pos = 0
    stop_loss = 0.0  # Dynamic stop-loss tracking
    dynamic_max_pos = MAX_POSITION_UNITS_CURRENT  # starting dynamic max pos
    N = len(live_trades)

    for i in range(N):
        tr = live_trades[i]
        if "label" not in tr:
            continue
        action = tr["label"]
        px = tr["price"]


        if pos_units != 0 and ((pos_units > 0 and px <= stop_loss) or (pos_units < 0 and px >= stop_loss)):
            tr["label"] = "PARTIAL_CLOSE"  # 
            action = "PARTIAL_CLOSE"
            pos_units = 0  # Reset position after forced closure
            avg_px = 0.0
            stop_loss = 0.0
            bars_in_pos = 0
            continue  # Move to next trade (skip further processing)


        if abs(pos_units) > 1e-8 and avg_px > 0:
            upnl_pct = (px - avg_px) / avg_px if pos_units > 0 else (avg_px - px) / avg_px
        else:
            upnl_pct = 0.0




        # If we're already long and see "OPEN_LONG_3", we might scale in or hold.
        # Minimal logic here for demonstration
        if action.startswith("OPEN_LONG_") and pos_units > 0:
            tr["label"] = "SCALE_IN_LONG" if upnl_pct >= scale_in_min_upnl else "HOLD"
            action = tr["label"]
        if action.startswith("OPEN_SHORT_") and pos_units < 0:
            tr["label"] = "SCALE_IN_SHORT" if upnl_pct >= scale_in_min_upnl else "HOLD"
            action = tr["label"]



        # *** Naive position simulation update (just so 'label' is consistent later) ***
        if action == "OPEN_LONG_1":
            pos_units += 1
            avg_px = px
            bars_in_pos = 0
        elif action == "OPEN_LONG_2":
            pos_units += 2
            avg_px = px
            bars_in_pos = 0
        elif action == "OPEN_LONG_3":
            pos_units += 3
            avg_px = px
            bars_in_pos = 0
        elif action == "OPEN_SHORT_1":
            pos_units -= 1
            avg_px = px
            bars_in_pos = 0
        elif action == "OPEN_SHORT_2":
            pos_units -= 2
            avg_px = px
            bars_in_pos = 0
        elif action == "OPEN_SHORT_3":
            pos_units -= 3
            avg_px = px
            bars_in_pos = 0
        elif action=="SCALE_IN_LONG":
            pos_units+=1
            avg_px= px
            bars_in_pos=0
        elif action=="SCALE_IN_SHORT":
            pos_units-=1
            avg_px= px
            bars_in_pos=0
        elif action == "PARTIAL_CLOSE":
            if pos_units > 0:
                pos_units -= 1
                if pos_units <= 0:
                    pos_units = 0
                    avg_px = 0
                    bars_in_pos = 0
            else:
                pos_units += 1
                if pos_units >= 0:
                    pos_units = 0
                    avg_px = 0
                    bars_in_pos = 0
        elif action == "CLOSE_FULL":
            if pos_units > 0:
                pos_units -= 1
                if pos_units <= 0:
                    pos_units = 0
                    avg_px = 0
                    bars_in_pos = 0
            else:
                pos_units += 1
                if pos_units >= 0:
                    pos_units = 0
                    avg_px = 0
                    bars_in_pos = 0
        elif action == "FLIP_LONG":
            pos_units = 1
            avg_px = px
            bars_in_pos = 0
        elif action == "FLIP_SHORT":
            pos_units = -1
            avg_px = px
            bars_in_pos = 0
        # TAKE_HALF, TAKE_QUARTER do not change the simulation position here
        # for demonstration unless you want them to.

        if abs(pos_units) > 1e-8:
            bars_in_pos += 1
        else:
            bars_in_pos = 0

def label_older_trades(bot, live_trades: List[dict], scale_in_min_upnl: float = 0.0):
    """
    1) Once the model is trained, assign OPEN_LONG/SHORT_x based on
       every remaining future bar (no fixed lookahead).
    2) Always run improved_label_simulation to enforce stops, scaling, etc.
    """
    N = len(live_trades)
    # need at least 2 bars to look ahead at all
    if N < 2:
        return


    # adaptive thresholds
    thr_strong, thr_moderate = bot.compute_label_thresholds(lookback=200)

    # for each bar where we can look ahead at least one bar
    for i in range(N - 1):
        tr = live_trades[i]
        if "label" in tr or tr.get("price", 0) <= 0:
            continue

        # take *all* the remaining future bars
        fut = live_trades[i+1:]
        if not fut:
            continue

        avgf = np.mean([x["price"] for x in fut])
        pct  = (avgf - tr["price"]) / tr["price"]

        if   pct >  thr_strong:   tr["label"] = "OPEN_LONG_3"
        elif pct >  thr_moderate:  tr["label"] = "OPEN_LONG_2"
        elif pct >  0:             tr["label"] = "OPEN_LONG_1"
        elif pct < -thr_strong:    tr["label"] = "OPEN_SHORT_3"
        elif pct < -thr_moderate:  tr["label"] = "OPEN_SHORT_2"
        elif pct <  0:             tr["label"] = "OPEN_SHORT_1"
        else:                      tr["label"] = "HOLD"

    # enforce stops, scaling, etc.
    improved_label_simulation(live_trades,  scale_in_min_upnl=0.002)



###############################################################################
# SUPER REFINED ML MODEL (Extended with RSI)
###############################################################################
# Corrected SuperRefinedML class with additional params for PositionManager and WaveComputer
class SuperRefinedML:
    def __init__(self,
                 logger: Logger,
                 hist_mgr: LongHistoryManager,
                 model_file=MODEL_FILE,
                 scaler_file=SCALER_FILE,
                 posmgr=None,            # Add posmgr (PositionManager)
                 wave_computer=None,
                 last_trade_time=None,
                 max_position_units_current=None):    # Add wave_computer (WaveComputer)
        self.logger = logger
        self.hist_mgr = hist_mgr
        self.model_file = model_file
        self.scaler_file = scaler_file
        self.last_trade_time = last_trade_time
        self.max_position_units_current = max_position_units_current

        if posmgr is None:
            raise ValueError("PositionManager (posmgr) must be provided")
        self.posmgr = posmgr   # Initialize posmgr (PositionManager)
        self.wave_computer = wave_computer  # Initialize wave_computer (WaveComputer)

        # Initialize RandomForest with more trees and better parameters
        self.clf = RandomForestClassifier(
            n_estimators=200,  # Increased number of estimators
            max_depth=441,      # Limit the depth of trees to avoid overfitting
            min_samples_leaf=32,  # Prevent overfitting on individual samples
            warm_start=True,
            random_state=95
        )
        self.model_trained = False
        self.scaler = StandardScaler()


    def load_model(self):
        if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
            try:
                self.clf = joblib.load(self.model_file)
                self.scaler = joblib.load(self.scaler_file)
                self.model_trained = True
                self.logger.log(f"[SuperRefinedML] Loaded from {self.model_file}")
            except Exception as e:
                self.logger.log(f"[SuperRefinedML.load_model] Error: {e}")

    def save_model(self):
        try:
            joblib.dump(self.clf, self.model_file)
            joblib.dump(self.scaler, self.scaler_file)
            self.logger.log(f"[SuperRefinedML] Saved model to {self.model_file}")
        except Exception as e:
            self.logger.log(f"[SuperRefinedML.save_model] Error: {e}")



    def train_model(self, df_websocket: pd.DataFrame, min_data: int = 100):
        """
        Trains on synthetic labeled data in `df_websocket` (live WebSocket data).
        Saves model+scaler when training succeeds.
        """
        self.logger.log("[DEBUG train_model] Called with df_websocket rows: "
                        f"{0 if df_websocket is None else len(df_websocket)}; min_data={min_data}")
    
        # 1) Validate synthetic data
        if df_websocket is None or df_websocket.empty:
            self.logger.log("[DEBUG train_model] No WebSocket data provided; skipping training.")
            return
        if len(df_websocket) < min_data:
            self.logger.log(f"[DEBUG train_model] Only {len(df_websocket)} rows (<{min_data}); skipping training.")
            return
    
        # ** Process the df_websocket (live data) separately **
    
        # Start with df_websocket (live WebSocket data)
        df_live = df_websocket.copy()
        self.logger.log(f"[DEBUG train_model] Using {len(df_live)} rows of live WebSocket data for training.")
    
        # ** Add PositionManager features to df_websocket (live data) **
        df_live["avg_entry_price"] = self.posmgr.avg_entry_price
        self.logger.log(f"[DEBUG] avg_entry_price (live): {self.posmgr.avg_entry_price}")
        
        df_live["bars_in_position"] = self.posmgr.bars_in_position
        self.logger.log(f"[DEBUG] bars_in_position (live): {self.posmgr.bars_in_position}")
        
        df_live["pos_units"] = self.posmgr.position_units
        self.logger.log(f"[DEBUG] pos_units (live): {self.posmgr.position_units}")
        
        df_live["pos_unrealized_pct"] = self.posmgr.get_unrealized_pct(df_live["price"].iloc[-1])  # last price
        self.logger.log(f"[DEBUG] pos_unrealized_pct (live): {self.posmgr.get_unrealized_pct(df_live['price'].iloc[-1])}")
        
        df_live["pos_realized_pnl"] = self.posmgr.realized_pnl
        self.logger.log(f"[DEBUG] realized_pnl (live): {self.posmgr.realized_pnl}")
        
        df_live["stop_loss"] = self.posmgr.stop_loss
        self.logger.log(f"[DEBUG] stop_loss (live): {self.posmgr.stop_loss}")
        
        df_live["max_position_units_current"] = self.posmgr.position_units  # Assuming max position units
        self.logger.log(f"[DEBUG] max_position_units_current (live): {self.posmgr.position_units}")
        
        df_live["buy_run_pnl"] = self.posmgr.buy_run_pnl  # Fetch buy_run_pnl from PositionManager
        self.logger.log(f"[DEBUG] buy_run_pnl (live): {self.posmgr.buy_run_pnl}")
        
        df_live["sell_run_pnl"] = self.posmgr.sell_run_pnl  # Fetch sell_run_pnl from PositionManager
        self.logger.log(f"[DEBUG] sell_run_pnl (live): {self.posmgr.sell_run_pnl}")
        
        # Assign the `side` as +1 for long, -1 for short (based on pos_units)
        df_live["side"] = np.where(self.posmgr.position_units > 0, 1, np.where(self.posmgr.position_units < 0, -1, 0))
        self.logger.log(f"[DEBUG] side (live): {np.where(self.posmgr.position_units > 0, 1, -1)}")
    
        # ** Feature Engineering for df_websocket (live data) **
        df_live['time_since_last_trade'] = (pd.Timestamp.now() - pd.to_datetime(self.last_trade_time)).total_seconds()
        df_live['time_of_day'] = df_live['timestamp'].apply(lambda x: pd.to_datetime(x, unit='s').hour)
        df_live['day_of_week'] = df_live['timestamp'].apply(lambda x: pd.to_datetime(x, unit='s').weekday())
    
        # ** Feature Engineering for additional statistical features **
        df_live["pchg"] = df_live["price"].pct_change().fillna(0)
        df_live["avg_pchg_10"] = df_live["pchg"].rolling(10).mean().fillna(0)
        df_live["std_pchg_10"] = df_live["pchg"].rolling(10).std().fillna(0)
        df_live["vol_10"] = df_live["qty"].rolling(10).sum().fillna(0)
    
        # ** Add the missing features here **
    
        # 1. `vol_avg_10` - average volume in the last 10 periods
        df_live['vol_avg_10'] = df_live['qty'].rolling(10).mean().fillna(0)
        
        # 2. `risk_reward` - risk reward ratio (example calculation)
        # 4. Interaction features: `pnl_stop_loss_interaction`, `pnl_risk_reward_interaction`
        # Calculate pnl_stop_loss_interaction based on the side
        df_live['pnl_stop_loss_interaction'] = np.where(
            df_live['side'] == 1,  # If long position (side == +1)
            df_live['buy_run_pnl'] * df_live['stop_loss'],  # Use buy_run_pnl for long positions
            np.where(
                df_live['side'] == -1,  # If short position (side == -1)
                df_live['sell_run_pnl'] * df_live['stop_loss'],  # Use sell_run_pnl for short positions
                0  # If flat position (side == 0), set to 0
            )
        )
        

        # 2. `risk_reward` - risk reward ratio (example calculation)
        df_live['risk_reward'] = np.where(
            df_live['side'] > 0,  # For long positions (side == +1)
            np.where(
                df_live['stop_loss'] != 0, 
                (df_live['price'] - df_live['avg_entry_price']) / (df_live['stop_loss'] - df_live['avg_entry_price']),
                0.0  # Default to 0 if stop_loss is 0 to avoid division by zero for long positions
            ),
            np.where(
                df_live['side'] < 0,  # For short positions (side == -1)
                np.where(
                    df_live['stop_loss'] != 0, 
                    (df_live['avg_entry_price'] - df_live['price']) / (df_live['avg_entry_price'] - df_live['stop_loss']),
                    0.0  # Default to 0 if stop_loss is 0 to avoid division by zero for short positions
                ),
                0.0  # For flat positions (side == 0), set risk_reward to 0
            )
        )

        
        # Calculate pnl_risk_reward_interaction based on the side
        df_live['pnl_risk_reward_interaction'] = np.where(
            df_live['side'] == 1,  # If long position (side == +1)
            df_live['buy_run_pnl'] * df_live['risk_reward'],  # Use buy_run_pnl for long positions
            np.where(
                df_live['side'] == -1,  # If short position (side == -1)
                df_live['sell_run_pnl'] * df_live['risk_reward'],  # Use sell_run_pnl for short positions
                0  # If flat position (side == 0), set to 0
            )
        )
        
        # 2. target_dist_pct based on the side (long or short)
        df_live['target_dist_pct'] = np.where(
            df_live['side'] > 0,  # For long positions (side == +1)
            (df_live['price'] - df_live['avg_entry_price']) / df_live['avg_entry_price'] * 100,
            np.where(
                df_live['side'] < 0,  # For short positions (side == -1)
                (df_live['avg_entry_price'] - df_live['price']) / df_live['avg_entry_price'] * 100, 
                0  # If flat position (side == 0), set to 0
            )
        )


        # 3. RSI (if compute_RSI is available)
        df_live['rsi'] = compute_RSI(df_live['price'])
        
        # 4. volatility regime
        df_live['vol_regime'] = (df_live['std_pchg_10'] > 0.01).astype(int)

        # 5. wave_score lag features
        df_live['wave_score_lag1'] = df_live['wave_score'].shift(1).fillna(method='bfill')
        df_live['wave_score_lag2'] = df_live['wave_score'].shift(2).fillna(method='bfill')


    
        # 3. `pos_realized_pnl_lag1` and `pos_realized_pnl_lag2` - lag features
        df_live['pos_realized_pnl_lag1'] = df_live['pos_realized_pnl'].shift(1).fillna(0)
        df_live['pos_realized_pnl_lag2'] = df_live['pos_realized_pnl'].shift(2).fillna(0)

        df_live['stop_loss_dist_pct'] = np.where(
            df_live['side'] > 0,  # For long positions (side == +1)
            (df_live['price'] - df_live['stop_loss']) / df_live['price'] * 100, 
            (df_live['stop_loss'] - df_live['price']) / df_live['price'] * 100  # For short positions (side == -1)
        )
    

        
        # 5. Cumulative `pos_realized_pnl` and `trade_count`
        df_live['cumulative_pos_realized_pnl'] = df_live['pos_realized_pnl'].cumsum().fillna(0)
        df_live['trade_count'] = df_live.groupby('side')['timestamp'].cumcount() + 1  # +1 to start from 1
        df_live['trade_count'] = df_live.groupby((df_live['side'] != df_live['side'].shift()).cumsum())['side'].cumcount() + 1


    
        # 6. `hist_ma_15` - historical moving average (15 periods)
        df_live['hist_ma_15'] = df_live['price'].rolling(window=15).mean().fillna(0)
        
        df_live.replace([np.inf, -np.inf], 1e10, inplace=True)
        df_live.fillna(0, inplace=True)
    
        # ** Feature Engineering End **
        #self.logger.log(f"[DEBUG train_model] DataFrame after feature engineering and fillna:\n{df_live.to_string()}")
    
        # 3) Define expected feature list for training
        feature_cols = [
            "price", "pchg", "avg_pchg_10", "std_pchg_10", "vol_10",
            "wave_score", "wave_score_lag1", "wave_score_lag2",
            "vol_regime", "bars_in_position", "pos_units", "pos_unrealized_pct",
            "pos_realized_pnl", "stop_loss", "time_of_day", "day_of_week",
            "time_since_last_trade", "max_position_units_current", "rsi",
            "vol_avg_10", "stop_loss_dist_pct", "target_dist_pct", "risk_reward",
            "pos_realized_pnl_lag1", "pos_realized_pnl_lag2",
            "pnl_stop_loss_interaction", "pnl_risk_reward_interaction",
            "cumulative_pos_realized_pnl", "trade_count",
            "avg_entry_price", "buy_run_pnl", "sell_run_pnl", "side", "hist_ma_15"
        ]
        
        self.logger.log(f"[DEBUG train_model] Expected features count: {len(feature_cols)}")
        
        # 4) Handle missing features in the df_websocket DataFrame (fill with 0)
        missing = [col for col in feature_cols if col not in df_live.columns]
        if missing:
            self.logger.log(f"[DEBUG train_model] Missing features detected: {missing}")
            for col in missing:
                df_live[col] = 0.0  # Fill missing features with zeros
            self.logger.log("[DEBUG train_model] Missing features filled with 0.0")
    
        # 5) Prepare X and y for model training
        X = df_live[feature_cols].values
        y = df_live["label"].values  # Assuming 'label' is the target for training
    
        self.logger.log(f"[DEBUG train_model] Prepared X.shape={X.shape}, y.shape={y.shape}")
    
        # 6) Scaling and training
        try:
            self.logger.log("[DEBUG train_model] Initializing MinMaxScaler…")
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.logger.log("[DEBUG train_model] Scaling complete. X_scaled.shape=" f"{X_scaled.shape}")
    
            self.logger.log("[DEBUG train_model] Initializing RandomForestClassifier…")
            self.clf = RandomForestClassifier(n_estimators=200, max_depth=441, min_samples_leaf=32, warm_start=True, random_state=95)
            self.clf.fit(X_scaled, y)
            self.logger.log("[DEBUG train_model] Model fitting complete.")
    
            # Save model and scaler
            self.save_model()
            self.logger.log(f"[DEBUG train_model] Saved model to {self.model_file}")
            self.logger.log(f"[DEBUG train_model] Saved scaler to {self.scaler_file}")
            self.model_trained = True
        except Exception as e:
            self.logger.log(f"[DEBUG train_model] Training failed with exception: {e}")




    def train_if_ready(self, live_trades, agg_interval, slow_agg_interval, logger, trade_agg_buffer, bot_ref):
        """
        Gathers synthetic labeled data, trains the model, and switches to slower aggregation after successful training.
        """
        # 1) Gather synthetic labeled data, but drop STOP_LOSS when flat
        synth_records = [
            t for t in live_trades
            if "label" in t
               and not (t["label"] in ("STOP_LOSS_TRIGGERED", "PARTIAL_CLOSE", "CLOSE_FULL") and t.get("pos_units", 0) == 0)
        ]
        df_synth = pd.DataFrame(synth_records) if synth_records else None

        if df_synth is not None:
            logger.log(f"[train_if_ready] Including {len(df_synth)} synthetic examples")

        # 2) Train the model using df_websocket (live data)
        self.train_model(
            df_websocket=df_synth,
            min_data=MIN_DATA_FOR_TRAIN
        )

        # 3) On first successful training, switch to slower aggregation bars
        if self.model_trained and agg_interval != slow_agg_interval:
            old = agg_interval
            bot_ref.agg_interval = slow_agg_interval
            trade_agg_buffer.clear()
            bot_ref.current_bucket = None
            logger.log(f"[train_if_ready] Warmup complete: switched from {old}s to {slow_agg_interval}s bars.")


    def predict_action(self, feats: dict) -> str:
        """
        Predict action based on input features using the trained model.
        Returns the predicted action as a string.
        """
        import numpy as np
        import pandas as pd
    
        # 1) Fallback if the model is not trained
        if not self.model_trained:
            self.logger.log("[predict_action] Model not trained. Defaulting to 'HOLD'.")
            return "HOLD"
    
        # 2) Full feature list must exactly match train_model’s list, plus any new fields
        X_cols = [
            "price", "pchg", "avg_pchg_10", "std_pchg_10", "vol_10",
            "wave_score", "wave_score_lag1", "wave_score_lag2",
            "vol_regime", "bars_in_position",
            "pos_units", "pos_unrealized_pct", "pos_realized_pnl",
            "stop_loss", "time_of_day", "day_of_week", "time_since_last_trade",
            "max_position_units_current", "rsi",
            "vol_avg_10", "stop_loss_dist_pct", "target_dist_pct", "risk_reward",
            "pos_realized_pnl_lag1", "pos_realized_pnl_lag2",
            "pnl_stop_loss_interaction", "pnl_risk_reward_interaction",
            "cumulative_pos_realized_pnl", "trade_count",
            "avg_entry_price", "buy_run_pnl", "sell_run_pnl", "side",
            "hist_ma_15"
        ]
        self.logger.log(f"[predict_action] Expecting features: {X_cols}")
    
        # 3) Ensure all necessary features exist, defaulting missing ones to 0.0
        missing = [c for c in X_cols if c not in feats]
        if missing:
            self.logger.log(f"[predict_action] Missing feats, defaulting to 0: {missing}")
            for c in missing:
                feats[c] = 0.0
    
        # 4) Log the raw feats dict
        self.logger.log(f"[predict_action] Features before building row: {feats}")
    
        # 5) Build the feature row, tracking any NaNs
        row = []
        nan_feats = []
        for c in X_cols:
            v = feats[c]
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1:
                v = v[0]
            if pd.isna(v):
                nan_feats.append(c)
                v = 0.0
            row.append(float(v))
        if nan_feats:
            self.logger.log(f"[predict_action] Replaced NaNs in: {nan_feats}")
    
        # 6) Log shape and contents of the row
        self.logger.log(f"[predict_action] Row shape: {len(row)}")
        self.logger.log(f"[predict_action] Row values: {row}")
    
        # 7) Scale & predict
        try:
            # Scaling features using the previously trained scaler
            Xs = self.scaler.transform([row])
            self.logger.log(f"[predict_action] Scaled features shape: {Xs.shape}")
    
            # Use the classifier to predict the action (whether to open long, short, or hold)
            prediction = self.clf.predict(Xs)[0]
            self.logger.log(f"[predict_action] Predicted action: {prediction}")
            return prediction
        except Exception as e:
            # If any error occurs during the prediction, log it and default to 'HOLD'
            self.logger.log(f"[predict_action] Error during prediction: {e}")
            return "HOLD"



###############################################################################
# TRADE STREAM
###############################################################################
import traceback

async def trade_stream(main_bot: "TraRyTrade_SelfSwimm_V1_SuperRefined"):
    url = f"wss://fstream.binance.com/ws/{varmove.Coin.lower()}@trade"
    while True:
        try:
            async with websockets.connect(url, ping_interval=3, ping_timeout=7) as ws:
                main_bot.logger.log(f"[trade_stream] Connected to {SYMBOL} trade stream.")
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    if data.get("e") == "trade" and data.get("s", "").upper() == SYMBOL:
                        try:
                            main_bot.on_new_trade(data)
                        except Exception:
                            main_bot.logger.log(
                                "[trade_stream] on_new_trade error:\n" +
                                traceback.format_exc()
                            )
        except Exception as e:
            main_bot.logger.log(f"[trade_stream] Connection error: {e}. Reconnecting in 0.1s.")
            await asyncio.sleep(0.1)




###############################################################################
# LIQUIDATION STREAM
###############################################################################
async def liquidation_stream(main_bot: "TraRyTrade_SelfSwimm_V1_SuperRefined"):
    """
    Monitors the forced liquidation stream. If a liquidation occurs,
    treat it similarly to a trade with "liq_events=1".
    """
    url = f"wss://fstream.binance.com/ws/{varmove.Coin.lower()}@forceOrder"
    while True:
        try:
            async with websockets.connect(url, ping_interval=3, ping_timeout=7) as ws:
                main_bot.logger.log(f"[liquidation_stream] Connected to {SYMBOL} liquidation stream.")
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    if data.get("e") == "forceOrder" and "o" in data:
                        o = data["o"]
                        px = float(o.get("ap", 0))
                        q = float(o.get("z", 0))
                        tms = float(o["T"]) / 1000.0
                        trade = {
                            "timestamp": tms, "price": px, "qty": q, 
                            "wave_score": 0.0, "liq_events": 1,
                            "e": "trade", "s": SYMBOL, "T": int(tms * 1000)
                        }
                        main_bot.on_new_trade(trade)
        except Exception as e:
            main_bot.logger.log(f"[liquidation_stream] Error: {e}. Reconnecting in 0.1s.")
            await asyncio.sleep(0.1)

###############################################################################
# REGISTRY SYNC
###############################################################################



def save_trades_to_file(file_path: str, trades: List[dict]):
    """
    Save the list of trades to disk.
    """
    try:
        with open(file_path, "w") as f:
            json.dump(trades, f, indent=2)
    except Exception as e:
        print(f"Error saving trades to {file_path}: {e}")


import json, os

def robust_write_json(file_path: str, data):
    """
    Write JSON to a temp file, then atomically rename it into place.
    This helps avoid partial/corrupt files if the process dies mid‑write.
    """
    temp_path = file_path + ".tmp"
    try:
        # 1. Write to the temp file
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()        # flush writes to OS buffers
            os.fsync(f.fileno())  # ensure it's really on disk (POSIX only)

        # 2. Atomically replace the real file with the temp
        os.replace(temp_path, file_path)
    except Exception as e:
        print(f"Error in robust_write_json({file_path}): {e}")
        # Optionally remove the temp file if still around
        if os.path.exists(temp_path):
            os.remove(temp_path)


###############################################################################
# MAIN BOT
###############################################################################
class TraRyTrade_SelfSwimm_V1_SuperRefined:
    def __init__(self, logger: Logger, posmgr: PositionManager):
        self.logger = logger
        self.name = SYMBOL
        self.binance_positions: Dict[str, dict] = {}
        self.client = Client(API_KEY, API_SECRET)

        # Fetch and normalize the Binance futures listenKey
        resp = self.client.futures_stream_get_listen_key()
        if isinstance(resp, dict):
            # older/newer wrappers return {"listenKey": "..."}
            self.listen_key = resp.get("listenKey")
        elif isinstance(resp, str):
            # some versions return the key directly
            self.listen_key = resp
        else:
            raise RuntimeError(f"Unexpected listenKey response type {type(resp)}: {resp!r}")

        if not self.listen_key:
            raise RuntimeError("Failed to fetch Binance listenKey")

        # Initialize the PositionManager and WaveComputer before passing to ML class
        self.posmgr = posmgr  # Pass PositionManager directly, already initialized
        self.wave_computer = AdvancedWaveComputer()

        # Initialize the LongHistoryManager
        self.hist_mgr = LongHistoryManager(self.client, SYMBOL, logger=self.logger)

        self.last_trade_time = time.time()

        self.max_position_units_current = MAX_POSITION_UNITS_CURRENT

        # Now pass initialized posmgr and wave_computer to the ML class
        self.model = SuperRefinedML(
            logger=self.logger,
            hist_mgr=self.hist_mgr,
            model_file=MODEL_FILE,
            scaler_file=SCALER_FILE,
            posmgr=self.posmgr,  # Pass PositionManager
            wave_computer=self.wave_computer,
            last_trade_time=self.last_trade_time,
            max_position_units_current = self.max_position_units_current
        )
        self.model.load_model()  # Load the ML model


        # ── Initialize other variables and settings ──
        self.live_trades: List[dict] = []  # This will store the live trades
        self.last_retrain_time = time.time()
        self.last_price = 0.0
        self.consecutive_count = 0
        self.last_model_action = None
        self.REQUIRED_CONSEC = 1
        self.max_position_units_current = MAX_POSITION_UNITS_CURRENT
        self.async_manager = TradingBotAsyncManager()

        # Aggregation intervals
        self.fast_agg_interval = 3    # use 5 s bars while model is cold
        self.slow_agg_interval = 6   # default 60 s bars
        self.agg_interval = self.fast_agg_interval
        self.current_bucket = None          # start time of current bucket
        self.trade_agg_buffer = []            # List[Tuple[price, qty]]

        # ── Load any prior state ──
        self.load_state()

        self.save_counter = 0
        self.last_save_time = time.time()

        self.API_error_ON = 0

        # ── Initial sync ──
        self.sync_pos_with_binance()


        


    def compute_label_thresholds(self, lookback: int = 50):
        """
        Use recent price returns to set:
          • strong move = 95th percentile
          • moderate move = 80th percentile
        """
        hist = self.hist_mgr.get_df()
        if len(hist) < lookback:
            # fallback to very small defaults until we have data
            return 0.001, 0.0003

        # compute pct-changes
        rets = hist['close'].pct_change().dropna().tail(lookback).abs()
        thr_strong   = float(np.percentile(rets, 95))
        thr_moderate = float(np.percentile(rets, 80))
        self.logger.log(f"[AdaptiveLabels] strong={thr_strong:.5f}, moderate={thr_moderate:.5f}")
        return thr_strong, thr_moderate


    


    async def _delayed_position_sync(self, delay=1.0):
        """
        Sleep briefly, then call sync_pos_with_binance().
        This gives the exchange time to fill the order before we fetch the new position.
        """
        await asyncio.sleep(delay)
        self.logger.log("[DelayedSync] Attempting position sync after fill.")
        self.sync_pos_with_binance()




    def load_trades_from_file(self, file_path: str) -> List[dict]:
        """
        Loads JSON data from a file and *forces* it to be a list of dicts.
        If anything is malformed or top-level is not a list, we return an empty list.
        """
        if not os.path.exists(file_path):
            return []
    
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading/parsing JSON from {file_path}: {e}")
            return []
    
        # If the top-level is already a list, check if each element is a dict. 
        if isinstance(data, list):
            # Filter out any non-dict items to ensure we only keep a list of dicts
            data = [x for x in data if isinstance(x, dict)]
            return data
    
        # If the top-level is a dict, check if there's a "trades" key 
        # that is itself a list
        if isinstance(data, dict):
            if "trades" in data and isinstance(data["trades"], list):
                # again filter out any non-dict items
                trades_list = [x for x in data["trades"] if isinstance(x, dict)]
                return trades_list
            else:
                # Otherwise, the dict doesn't match the expected shape
                # We'll just log it and return empty
                print(f"[load_trades_from_file] Expected a list, got a dict. Returning empty.")
                return []
    
        # If it’s neither a list nor a dict, just return empty
        return []



    def load_state(self):
        """
        Load saved trades and position if present.
        Ensures self.live_trades is always a list of dicts.
        """
        if os.path.exists(TRADES_FILE):
            trades = self.load_trades_from_file(TRADES_FILE)
            if not isinstance(trades, list):
                self.logger.log(f"[MainBot] TRADES_FILE not a list. Resetting to empty.")
                trades = []
            # Finally, store only the validated list
            self.live_trades = trades
            label_older_trades(self, self.live_trades, scale_in_min_upnl=0.002)
            self.logger.log(f"[MainBot] Loaded {len(self.live_trades)} trades from file.")
        else:
            self.live_trades = []
    
        # Position loading
        if os.path.exists(POSITION_FILE):
            try:
                with open(POSITION_FILE, "r") as f:
                    data = json.load(f)
                self.posmgr.from_dict(data)
            except Exception as e:
                self.logger.log(f"[MainBot] Error loading position: {e}")
    
        # Make sure your model is loaded
        self.model.load_model()




    def save_trades_to_file(self, file_path: str, trades: List[dict]):
        """
        Safely save the list of trades to disk via our robust writer.
        """
        try:
            robust_write_json(file_path, trades)
        except Exception as e:
            print(f"Error saving trades to {file_path}: {e}")
    
    def save_state(self):
        """
        Periodically or after each trade, save state to disk (robustly).
        """
        try:
            self.save_trades_to_file(TRADES_FILE, self.live_trades)
        except Exception as e:
            self.logger.log(f"[MainBot] Error saving trades: {e}")
    
        try:
            robust_write_json(POSITION_FILE, self.posmgr.to_dict())
        except Exception as e:
            self.logger.log(f"[MainBot] Error saving position: {e}")


    async def _keepalive_listen_key(self):
        """
        Every 30 min extend the user‑data stream so the listenKey doesn't expire.
        """
        while True:
            try:
                self.client.futures_stream_keepalive(self.listen_key)
                self.logger.log("[UserData] listenKey keep‑alive sent")
            except Exception as e:
                self.logger.log(f"[UserData][ERROR] keepalive failed: {e}")
            await asyncio.sleep(30 * 60)


    async def _listen_user_data(self):
        """
        Connect to Binance Futures User Data WebSocket to listen for position updates
        and immediately populate positions for the specified symbol.
        """
        url = f"wss://fstream.binance.com/ws/{self.listen_key}"
    
        try:
            self.logger.log(f"[UserData] ▶ Connecting to user data stream...")
            async with websockets.connect(url, ping_interval=30, ping_timeout=10) as ws:
                self.logger.log(f"[UserData] ✔ Connected to {url}")
    
                while True:
                    raw = await ws.recv()
                    data = json.loads(raw)
    
                    self.logger.log(f"[UserData][RECV] {data}")
    
                    # Handle the 'ACCOUNT_UPDATE' event to update position data
                    if data.get("e") == "ACCOUNT_UPDATE":
                        for pos in data["a"]["P"]:
                            symbol = pos.get("s")
                            if symbol == SYMBOL:
                                self.binance_positions[symbol] = pos
                                self.logger.log(f"[UserData] Updated position for {symbol}: {pos}")
        except Exception as e:
            self.logger.log(f"[UserData][ERROR] {e!r}. Reconnecting…")
            # If connection fails, fall back to REST to fetch positions
            await self.sync_pos_with_binance()



    def fetch_binance_positions(self) -> Dict[str, dict]:
        """
        Return the most recent futures positions received via the user data stream.
        """
        return self.binance_positions


    def sync_pos_with_binance(self):
        """
        Sync PositionManager from Binance. First tries the user-data stream cache;
        if no entry is found for SYMBOL, falls back to a REST call.
        Supports both full and abbreviated field names.
        """

    
        # 1) Try user-data stream cache
        #all_pos = self.fetch_binance_positions()
        #self.logger.log(f"[SyncBinance][DEBUG] binance_positions keys: {list(all_pos.keys())}")
        #info = all_pos.get(SYMBOL)
    
        # 2) If empty, fallback to REST
        if 1==1:
            self.logger.log(f"[SyncBinance] ⚠ No entry for symbol `{SYMBOL}` in stream cache, falling back to REST.")
            try:
                resp = self.client.futures_position_information(symbol=SYMBOL)
                # futures_position_information returns a list of dicts: pick the first
                if isinstance(resp, list) and resp:
                    info = resp[0]
                    self.logger.log(f"[SyncBinance][DEBUG] REST position info for {SYMBOL}: {info}")
                else:
                    self.logger.log(f"[SyncBinance] ⚠ REST call returned no position info for {SYMBOL}.")
                    return
            except Exception as e:
                self.logger.log(f"[SyncBinance][ERROR] REST fallback failed: {e}")
                return
        else:
            self.logger.log(f"[SyncBinance][DEBUG] stream position info for {SYMBOL}: {info}")
    
        # 3) Parse fields (full or abbreviated)
        try:
            raw_amt   = info.get("positionAmt", info.get("pa"))
            raw_price = info.get("entryPrice",  info.get("ep"))
            self.logger.log(f"[SyncBinance][DEBUG] raw positionAmt={raw_amt}, entryPrice={raw_price}")
    
            amt         = float(raw_amt)
            entry_price = float(raw_price)
            self.logger.log(f"[SyncBinance][DEBUG] parsed amt={amt:.6f}, entry_price={entry_price:.6f}")
        except Exception as e:
            self.logger.log(f"[SyncBinance][ERROR] parsing fields failed: {e}")
            return
    
        # 4) Compute units
        try:
            units = int(amt / TRADE_UNIT_SIZE)
            self.logger.log(f"[SyncBinance][DEBUG] computed units = int({amt} / {TRADE_UNIT_SIZE}) = {units}")
        except Exception as e:
            self.logger.log(f"[SyncBinance][ERROR] converting amt→units failed: {e}")
            return
    
        # 5) Log old vs new
        old_units = self.posmgr.position_units
        old_avg   = self.posmgr.avg_entry_price
        self.logger.log(f"[SyncBinance] ─ Old state → units={old_units}, avg_entry_price={old_avg:.6f}")
        self.logger.log(f"[SyncBinance] ─ New state → amt={amt:.6f}, units={units}, entry_price={entry_price:.6f}")
    
        # 6) Update PositionManager
        self.posmgr.position_units  = units
        self.posmgr.avg_entry_price = entry_price
        self.posmgr.update_side_from_units()
        self.logger.log(f"[SyncBinance][DEBUG] PositionManager side set to {self.posmgr.side}")
    

        # 8) Final verification
        self.logger.log(
            f"[SyncBinance] ✔ Completed sync: "
            f"units={self.posmgr.position_units}, "
            f"avg_entry={self.posmgr.avg_entry_price:.6f}, "
            f"stop_loss={self.posmgr.stop_loss:.6f}"
        )




    def on_new_trade(self, data: dict):
        # ensure live_trades is always a list
        if not isinstance(self.live_trades, list):
            self.logger.log("self.live_trades is not a list. Reset to [].")
            self.live_trades = []
    
        # parse incoming tick (handle both raw and our own replayable format)
        px  = float(data.get("p", data.get("price", 0)))
        qty = float(data.get("q", data.get("qty", 0)))
        tms = float(data.get("T", data.get("timestamp", 0))) / 1000.0  # seconds
    
        # record for final PnL/logging and update wave buffer
        self.last_price = px
        self.wave_computer.add_trade(px, qty)
    
        # feed into long-history manager
        self.hist_mgr.append_trade(px, qty, tms)
    
        # ——— BUCKET AGGREGATION ———
        bucket = int(tms // self.agg_interval) * self.agg_interval
        if self.current_bucket is None:
            self.current_bucket = bucket
    
        # when we move into a new bucket, flush the previous one
        if bucket != self.current_bucket and self.trade_agg_buffer:
            total_qty = sum(t["qty"] for t in self.trade_agg_buffer)
            vwap      = sum(t["price"] * t["qty"] for t in self.trade_agg_buffer) / total_qty
            ws        = self.wave_computer.compute_wave_score()
            le        = sum(t.get("liq_events", 0) for t in self.trade_agg_buffer)
    
            agg = {
                "timestamp":  self.current_bucket,
                "price":      vwap,
                "qty":        total_qty,
                "wave_score": float(ws),
                "liq_events": int(le),
            }
    
            self.live_trades.append(agg)
    
            # cap memory
            if len(self.live_trades) > 12000:
                self.live_trades = self.live_trades[-12000:]
    
            # reset for next bucket
            self.trade_agg_buffer.clear()
            self.current_bucket = bucket
    
        # always buffer the raw tick as a dict
        raw = {
            "price":      px,
            "qty":        qty,
            "wave_score": 0.0,
            "liq_events": data.get("liq_events", 0),
        }
        self.trade_agg_buffer.append(raw)
        # ————— END AGGREGATION —————
    
        # label / train / infer / trade / bookkeeping
        label_older_trades(self, self.live_trades, scale_in_min_upnl=0.002)

        # Periodically retrain the model
        if time.time() - self.last_retrain_time >= RETRAIN_INTERVAL_SEC:
            self.logger.log("[MainBot] Triggering train_if_ready()")
            self.model.train_if_ready(
                live_trades=self.live_trades,
                agg_interval=self.agg_interval,
                slow_agg_interval=self.slow_agg_interval,
                logger=self.logger,
                trade_agg_buffer=self.trade_agg_buffer,
                bot_ref=self  # so it can update agg_interval + current_bucket
            )
            self.last_retrain_time = time.time()

            self.last_retrain_time = time.time()


        self.do_inference_and_trade()
        self.posmgr.increment_bars_in_position()
    
        # periodic save
        self.save_counter += 1
        now = time.time()
        if self.save_counter >= 50 or (now - self.last_save_time > 30):
            self.save_state()
            self.save_counter = 0
            self.last_save_time = now
    
        # profit‐alert
        unreal_pct = self.posmgr.get_unrealized_pct(px)
        if unreal_pct >= PROFIT_THRESHOLD:
            self.logger.log(f"[ProfitIndicator] Profitable trade: {unreal_pct*100:.2f}% profit!")
    
        # debug dump
        self.logger.log(
            f"\n================= POSITION INFO =================\n"
            f"Coin: {varmove.Coin.upper()}\n"
            f"Trade Amt: {TRADE_UNIT_SIZE:.6f}\n"
            f"Max Pos (default): {MAX_POSITION_UNITS_ABS:.2f}\n"
            f"Max Pos (current internal): {self.max_position_units_current:.2f}\n"
            f"Current Price: {px:.6f}\n"
            f"Pos Units: {self.posmgr.position_units:.2f}\n"
            f"Avg Entry: {self.posmgr.avg_entry_price:.6f}\n"
            f"Stop Loss: {self.posmgr.stop_loss:.6f}\n"
            f"Bars in Pos: {self.posmgr.bars_in_position}\n"
            f"Unrealized PnL: {self.posmgr.get_unrealized_pnl(px):.6f}\n"
            f"Realized PnL: {self.posmgr.realized_pnl:.6f}\n"
            f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"==================================================\n"
        )







    def do_inference_and_trade(self):
        now = time.time()
    
        # 1) Throttle so we don’t spam too many orders
        if now - self.last_trade_time < MIN_COOLDOWN_SEC_MAIN:
            return
    
        # 2) Build a DataFrame from recent aggregated bars
        trades_df = pd.DataFrame(self.live_trades).tail(5000)
        if trades_df.empty:
            return
    
        # 3) Compute rolling‐window features (all NaNs filled)
        trades_df["pchg"] = trades_df["price"].pct_change().fillna(0)
        trades_df["avg_pchg_10"] = trades_df["pchg"].rolling(10).mean().fillna(0)
        trades_df["std_pchg_10"] = trades_df["pchg"].rolling(10).std().fillna(0)
        trades_df["vol_10"] = trades_df["qty"].rolling(10).sum().fillna(0)
        trades_df["rsi"] = compute_RSI(trades_df["price"])
    
        # Ensure wave_score column exists and fill any NaNs with 0.0
        if "wave_score" not in trades_df.columns:
            trades_df["wave_score"] = 0.0
        else:
            trades_df["wave_score"] = trades_df["wave_score"].fillna(0.0)
    
        trades_df["wave_score_lag1"] = trades_df["wave_score"].shift(1).fillna(0.0)
        trades_df["wave_score_lag2"] = trades_df["wave_score"].shift(2).fillna(0.0)
    
        trades_df["vol_regime"] = (trades_df["std_pchg_10"] > 0.01).astype(int)
        trades_df["vol_avg_10"] = trades_df["qty"].rolling(10).mean().fillna(0)
    
        # 4) Grab the latest bar
        last = trades_df.iloc[-1]
        px = float(last["price"])
    
        # 5) STOP-LOSS override
        forced_stoploss = False
        action = "HOLD"
        stop = self.posmgr.stop_loss
        if not self.posmgr.is_flat():
            if self.posmgr.is_long() and px <= stop:
                self.logger.log(f"[STOPLOSS] Price {px:.6f} ≤ stop {stop:.6f}, forcing PARTIAL_CLOSE.")
                action = "PARTIAL_CLOSE"
                forced_stoploss = True
            elif self.posmgr.is_short() and px >= stop:
                self.logger.log(f"[STOPLOSS] Price {px:.6f} ≥ stop {stop:.6f}, forcing PARTIAL_CLOSE.")
                action = "PARTIAL_CLOSE"
                forced_stoploss = True
    
        # 6) ML inference (if not forced stop)
        if not forced_stoploss:
            # 6a) compute hist_ma_15 exactly as in engineer_features ← UPDATED
            hist_df = self.hist_mgr.get_df()
            if not hist_df.empty:
                feats_hist_ma_15 = float(
                    hist_df['close']
                        .rolling(15)
                        .mean()
                        .iloc[-1]
                )
            else:
                feats_hist_ma_15 = 0.0
    
            # 6b) Assemble feature dict (no NaNs), including hist_ma_15 ← UPDATED
            feats = {
                "price": px,
                "pchg": float(last["pchg"]),
                "avg_pchg_10": float(last["avg_pchg_10"]),
                "std_pchg_10": float(last["std_pchg_10"]),
                "vol_10": float(last["vol_10"]),
                "wave_score": float(last["wave_score"]),
                "wave_score_lag1": float(last["wave_score_lag1"]),
                "wave_score_lag2": float(last["wave_score_lag2"]),
                "vol_regime": int(last["vol_regime"]),
                "bars_in_position": int(self.posmgr.bars_in_position),
                "pos_units": float(self.posmgr.position_units),
                "pos_unrealized_pct": float(self.posmgr.get_unrealized_pct(px)),
                "pos_realized_pnl": float(self.posmgr.realized_pnl),
                "stop_loss": float(self.posmgr.stop_loss),
                "time_of_day": datetime.datetime.utcfromtimestamp(now).hour
                              + datetime.datetime.utcfromtimestamp(now).minute/60,
                "day_of_week": datetime.datetime.utcfromtimestamp(now).weekday(),
                "time_since_last_trade": now - self.last_trade_time,
                "max_position_units_current": float(self.max_position_units_current),
                "rsi": float(last["rsi"]),
                "vol_avg_10": float(last["vol_avg_10"]),
                "stop_loss_dist_pct": abs(px - self.posmgr.stop_loss) / px if px else 0.0,
                "target_dist_pct": abs(feats_hist_ma_15 - px) / px if px else 0.0,
                "risk_reward": (
                    (abs(feats_hist_ma_15 - px)/px) /
                    (abs(px - self.posmgr.stop_loss)/px + 1e-6)
                ) if px else 0.0,
                "avg_entry_price": float(self.posmgr.avg_entry_price),
                "buy_run_pnl": float(self.posmgr.buy_run_pnl),
                "sell_run_pnl": float(self.posmgr.sell_run_pnl),
                "side": int(np.sign(self.posmgr.position_units)),
                "hist_ma_15": feats_hist_ma_15
            }
    
            # 7) Inference and action decision
            action = self.model.predict_action(feats)
            if action in ("PARTIAL_CLOSE", "CLOSE_FULL", "STOP_LOSS_TRIGGERED") and self.posmgr.is_flat():
                action = "HOLD"



        # 7) Record forced stop trades (only when we actually had a position)
        if forced_stoploss and not self.posmgr.is_flat():
            self.live_trades.append({
                "timestamp": now,
                "price":     px,
                "label":     "STOP_LOSS_TRIGGERED"
            })

        # 8) Breakeven-move logic
        if not forced_stoploss and not self.posmgr.is_flat():
            be_moved = (
                self.posmgr.stop_loss >= self.posmgr.avg_entry_price
                if self.posmgr.is_long()
                else self.posmgr.stop_loss <= self.posmgr.avg_entry_price
            )
            if not be_moved and self.posmgr.get_unrealized_pct(px) >= MOVE_STOP_BREAKEVEN_level:
                action = "MOVE_STOP_BREAKEVEN"

        # 9) Require consecutive signals
        if action == self.last_model_action:
            self.consecutive_count += 1
        else:
            self.consecutive_count     = 1
            self.last_model_action     = action
        if self.consecutive_count < self.REQUIRED_CONSEC:
            action = "HOLD"

        # 10) Sync & execute
        if action != "HOLD":
            self.logger.log("[SYNC] Checking position before executing trade...")
            self.sync_pos_with_binance()

        self.logger.log(
            f"[MainBot] Final action: {action}, price={px:.5f}, "
            f"pos_units={self.posmgr.position_units:.2f}, "
            f"maxPos={self.max_position_units_current}"
        )
        prev_units = self.posmgr.position_units
    

        # ---------------
        # Trade execution
        # ---------------
        sign = 0

        if action.startswith("OPEN_LONG_"):
            suffix = action.split("_")[-1]
            if suffix == "1":
                sign = 1
            elif suffix == "2":
                sign = 2
            elif suffix == "3":
                sign = 3
            else:
                sign = 0  

        elif action.startswith("OPEN_SHORT_"):
            suffix = action.split("_")[-1]
            if suffix == "1":
                sign = -1
            elif suffix == "2":
                sign = -2
            elif suffix == "3":
                sign = -3
            else:
                sign = 0








        elif action == "CLOSE_FULL" and not self.posmgr.is_flat():
            abs_units = abs(self.posmgr.position_units)     
            if abs_units > 10000:
                step_size = max(1, abs_units // 100)  # Close 0.03% each 
            elif abs_units > 5000:
                step_size = max(1, abs_units // 200)  # Close 0.04% each 
            elif abs_units > 2000:
                step_size = max(1, abs_units // 300)  # Close 0.001% each 
            elif abs_units > 12:
                step_size = max(1, abs_units // 20)  # Close 0.001% each 
            elif abs_units > 4:
                step_size = max(1, abs_units // 6)  # Close 0.001% each 
            elif abs_units > 2:
                step_size = max(1, abs_units // 2)  # Close 0.001% each 
            else:
                step_size = 1  

            if self.posmgr.is_long():
                sign = -step_size 
            else:
                sign = +step_size 

            


        elif action == "PARTIAL_CLOSE" and not self.posmgr.is_flat():
            abs_units = abs(self.posmgr.position_units)     
            if abs_units > 10000:
                step_size = max(1, abs_units // 100)  # Close 0.03% each 
            elif abs_units > 5000:
                step_size = max(1, abs_units // 200)  # Close 0.04% each 
            elif abs_units > 2000:
                step_size = max(1, abs_units // 300)  # Close 0.001% each 
            elif abs_units > 12:
                step_size = max(1, abs_units // 20)  # Close 0.001% each 
            elif abs_units > 4:
                step_size = max(1, abs_units // 6)  # Close 0.001% each 
            elif abs_units > 2:
                step_size = max(1, abs_units // 2)  # Close 0.001% each 
            elif abs_units == 1:
                step_size = 1            
            else:
                step_size = 1  
            
            
            if self.posmgr.is_long():
                sign = -step_size 
            else:
                sign = +step_size 

            
            #sign = -1 if self.posmgr.is_long() else 1
            
            




        elif action == "FLIP_LONG":
            if not self.posmgr.is_flat():
                sign = 1
        elif action == "FLIP_SHORT":
            if not self.posmgr.is_flat():
                sign = -1


        elif  action == "SCALE_IN_LONG":
            if  abs(self.posmgr.position_units + 1) <= self.max_position_units_current:
                sign = 1
                    
                
        elif action == "SCALE_IN_SHORT":
            if  abs(self.posmgr.position_units + 1) <= self.max_position_units_current:
                sign = -1



        elif action == "TAKE_HALF" and not self.posmgr.is_flat():
            if self.posmgr.is_long():
                sign = -int(math.floor(self.posmgr.position_units / 2))
            else:
                sign = int(math.floor(abs(self.posmgr.position_units) / 2))
        elif action == "TAKE_QUARTER" and not self.posmgr.is_flat():
            if self.posmgr.is_long():
                sign = -int(math.floor(self.posmgr.position_units / 4))
            else:
                sign = int(math.floor(abs(self.posmgr.position_units) / 4))
        elif action == "MOVE_STOP_BREAKEVEN":
            self.posmgr.move_stop_to_breakeven()
            
            
  
        elif action.startswith("ADJUST_MAX_POS_"):
            newmax = int(action.split("_")[-1])
            if 1 <= newmax <= MAX_POSITION_UNITS_ABS:
                self.logger.log(f"[{self.name}] ML adjusts max pos to {newmax}")
                self.max_position_units_current = newmax

        if self.posmgr.position_units  == 0:
            if abs(sign) > self.max_position_units_current:
                sign = 0
            else:
                sign = sign        
        elif self.posmgr.is_long() and sign > 0 and ( self.posmgr.position_units + sign ) < self.max_position_units_current: 
            sign = sign
        elif self.posmgr.is_long() and sign < 0 :
            sign = sign
            
        elif self.posmgr.is_short() and sign < 0 and ( abs(self.posmgr.position_units)  + abs(sign) )  < self.max_position_units_current: 
            sign = sign
        elif self.posmgr.is_short() and sign > 0 :
            sign = sign
        else:
            sign = 0     
              
        

        # If sign is still nonzero, we submit the order
        if sign != 0:
            self.submit_market_order("BUY" if sign > 0 else "SELL", px, sign, action)
            self.last_trade_time = now





    def submit_market_order(self, side: str, fill_price: float, lots: float, action: str):
        """
        Create and send a market order via Binance futures if REAL_TRADING is True.
        Then adjust local PositionManager as if the fill were immediate.
        """
        # Compute quantity in base units
        qty = float(abs(lots) * TRADE_UNIT_SIZE)
        if qty <= 0:
            return

        prev_units = self.posmgr.position_units
        self.logger.log(
            f"[MainBot] Submitting Binance order: side={side}, lots={lots}, qty={qty:.6f}, price={fill_price:.5f}"
        )

        if not REAL_TRADING:
            return

        try:
            # 1) Place the market order
            resp = self.client.futures_create_order(
                symbol=SYMBOL,
                side=side,
                type="MARKET",
                quantity=qty
            )
            self.logger.log(f"[Binance] Order response: {resp}")

            # 2) Locally adjust the position manager
            delta = lots if side == "BUY" else -lots
            self.posmgr.adjust_position(delta, fill_price)

            # 3) Reapply stop‑loss if it wasn’t already moved to breakeven
            if self.posmgr.is_long() and not self.posmgr.stop_moved_to_breakeven:
                self.posmgr.stop_loss = self.posmgr.avg_entry_price * (1.0 - STOP_LOSS_ML)
            elif self.posmgr.is_short() and not self.posmgr.stop_moved_to_breakeven:
                self.posmgr.stop_loss = self.posmgr.avg_entry_price * (1.0 + STOP_LOSS_ML)

            new_units = self.posmgr.position_units
            # 4) Log ML decision asynchronously
            asyncio.create_task(async_log_ml_decision(action, fill_price, prev_units, new_units, True))

            self.logger.log(f"[MainBot] Order executed. New pos_units: {new_units:.2f}")

        except Exception as e:
            self.logger.log(f"[MainBot] Binance order error: {e}")
            # ML decision = failed
            asyncio.create_task(async_log_ml_decision(action, fill_price, prev_units, self.posmgr.position_units, False))




    def get_performance(self) -> dict:
        upnl = self.posmgr.get_unrealized_pnl(self.last_price)
        rpnl = self.posmgr.realized_pnl
        return {
            "total_pnl": upnl + rpnl,
            "unrealized_pnl": upnl,
            "realized_pnl": rpnl,
            "pos_units": self.posmgr.position_units
        }





###############################################################################
# MAIN APPLICATION CLASS
###############################################################################
class TraRyMainSuperRefined:
    def __init__(self):
        self.logger = Logger()
        
        # Instantiate PositionManager before passing to TraRyTrade_SelfSwimm_V1_SuperRefined
        posmgr = PositionManager(
            trade_unit_size=TRADE_UNIT_SIZE,
            stop_loss_pct=STOP_LOSS_ML,
            logger=self.logger
        )
        
        # Pass the posmgr to the TraRyTrade_SelfSwimm_V1_SuperRefined constructor
        self.bot = TraRyTrade_SelfSwimm_V1_SuperRefined(self.logger, posmgr)

    async def periodic_binance_sync(self):
        while True:
            try:
                self.logger.log("[MainApp] Running periodic Binance position sync...")
                self.bot.sync_pos_with_binance()
            except Exception as e:
                self.logger.log(f"[MainApp] Binance sync error: {e}")
            await asyncio.sleep(1.0)


    async def periodic_log_flusher(self):
        # Flush logs every few seconds
        while True:
            await asyncio.sleep(3)
            self.logger.flush(force=True)



    async def run(self):
        try:
            await asyncio.gather(
                trade_stream(self.bot),
                liquidation_stream(self.bot),
                self.periodic_binance_sync(),
                self.periodic_log_flusher()  
            )
        except KeyboardInterrupt:
            self.logger.log("[MainApp] KeyboardInterrupt received. Exiting...")
        finally:
            self.bot.save_state()
            self.logger.flush(force=True)
            self.bot.hist_mgr._persist()

async def main():
    app = TraRyMainSuperRefined()
    await app.run()



###############################################################################
# MAIN ENTRY POINT
###############################################################################
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting.")
