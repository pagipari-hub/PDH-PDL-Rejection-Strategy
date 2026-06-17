"""
PDH/PDL Rejection Strategy — Backtest Engine
=============================================
Strategy logic:
  - Universe: Nifty 200 stocks
  - Timeframe: 5-minute candles
  - Entry window: 09:15 - 10:30 IST only
  - Setup (LONG, at PDL):
      1. Rejection candle: wick touches/pierces PDL, candle CLOSES above PDL
      2. A later candle crosses VWAP from below to above and CLOSES above VWAP
      3. Entry = that VWAP-cross candle's close + Rs 0.10 (limit order)
      4. SL = PDL (the level itself)
      5. Skip trade if achievable RR < 1:1.5
      6. Target 1 = 1:2 RR
      7. After Target 1 hit: switch to trailing stop = last confirmed 5-candle
         swing LOW (fixed reference until a new swing low forms)
      8. Exit when price closes below the trailing swing low
  - Setup (SHORT, at PDH): mirror image of the above
  - Constraints:
      - Max 1 trade per stock per day
      - Max 3 concurrent open positions across the whole portfolio
      - Position sizing: fixed risk per trade (RISK_PER_TRADE), 
        quantity = risk / |entry - SL|

Data source: Angel One SmartAPI (historical candle API)
Credentials are read from environment variables (set as GitHub Actions secrets):
      ANGEL_CLIENT_ID
      ANGEL_API_KEY
      ANGEL_TOTP_SECRET
      ANGEL_MPIN        (or ANGEL_PASSWORD, depending on your login mode)

Output: console summary (win rate, total P&L, avg RR achieved) + a CSV of all
trades for further analysis (saved to trade_log.csv).
"""

import os
import time
import json
import math
import pyotp
import logging
import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from SmartApi import SmartConnect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("pdh_pdl_backtest")


# ============================================================================
# CONFIG
# ============================================================================

class Config:
    # --- Angel One credentials (from env vars / GitHub Actions secrets) ---
    CLIENT_ID = os.environ.get("ANGEL_CLIENT_ID", "")
    API_KEY = os.environ.get("ANGEL_API_KEY", "")
    TOTP_SECRET = os.environ.get("ANGEL_TOTP_SECRET", "")
    MPIN = os.environ.get("ANGEL_MPIN", "")  # numeric PIN used for login

    # --- Backtest period ---
    # Angel One historical API allows much longer 5-min history than yfinance.
    # Keep this conservative to start; widen once the pipeline is verified.
    LOOKBACK_DAYS = 180

    # --- Strategy parameters ---
    ENTRY_WINDOW_START = dt.time(9, 15)
    ENTRY_WINDOW_END = dt.time(10, 30)
    MARKET_CLOSE = dt.time(15, 30)

    ENTRY_BUFFER = 0.10          # Rs 0.10 (2 tick) buffer on entry limit price
    MIN_RR = 1.5                 # skip trade if achievable RR < 1:1.5
    TARGET_RR = 2.0              # Target 1 = 1:2 RR
    SWING_LOOKBACK = 5           # 5-candle swing structure for trailing
    MAX_CONCURRENT_POSITIONS = 3
    MAX_TRADES_PER_STOCK_PER_DAY = 1

    RISK_PER_TRADE = 1000.0      # Rs 1000 fixed risk per trade

    CANDLE_INTERVAL = "FIVE_MINUTE"   # Angel One interval code
    EXCHANGE = "NSE"

    # --- Historical fetch chunking ---
    # Angel One's documented limit for FIVE_MINUTE interval is ~30 days per
    # request (longer intervals allow more, shorter intervals allow less).
    # We use 30 to minimize total request count.
    CHUNK_DAYS = 30

    # --- Rate-limit friendliness for Angel One historical API ---
    REQUEST_SLEEP_SEC = 3.5      # pause between every request, regardless of outcome
    MAX_RETRIES = 8              # retries specifically for rate-limit / transient errors
    RETRY_BACKOFF_SEC = 5.0      # base backoff, multiplied by attempt number (linear backoff)
    RATE_LIMIT_SLEEP_SEC = 45    # extra sleep specifically when a rate-limit message is detected
    RATE_LIMIT_MARKERS = (
        "exceeding access rate",
        "access denied",
        "rate limit",
        "too many requests",
    )

    # --- Caching ---
    CANDLE_CACHE_DIR = "candle_cache"
    INSTRUMENT_MASTER_CACHE_PATH = "instrument_master.csv"
    INSTRUMENT_MASTER_MAX_AGE_HOURS = 24  # refresh once per day


# ============================================================================
# NIFTY 200 UNIVERSE
# ============================================================================
# Static snapshot of Nifty 200 constituents (NSE trading symbols, no suffix).
# Index rebalances semi-annually (Mar/Sep) -- update this list periodically.
# Source: NSE Indices Nifty 200 constituent list.

NIFTY_200_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "BHARTIARTL", "ICICIBANK", "INFY", "SBIN",
    "LICI", "HINDUNILVR", "ITC", "LT", "BAJFINANCE", "HCLTECH", "MARUTI",
    "KOTAKBANK", "SUNPHARMA", "AXISBANK", "M&M", "ULTRACEMCO", "NTPC",
    "ADANIENT", "TITAN", "BAJAJFINSV", "ONGC", "ADANIPORTS", "WIPRO",
    "ASIANPAINT", "POWERGRID", "COALINDIA", "NESTLEIND", "JSWSTEEL",
    "BEL", "TATAMOTORS", "GRASIM", "TRENT", "TECHM", "HINDALCO", "SBILIFE",
    "HDFCLIFE", "CIPLA", "TATASTEEL", "DRREDDY", "EICHERMOT", "APOLLOHOSP",
    "INDUSINDBK", "BAJAJ-AUTO", "BRITANNIA", "DIVISLAB", "SHREECEM",
    "VEDL", "PIDILITIND", "GODREJCP", "HAVELLS", "DABUR", "SIEMENS",
    "ADANIGREEN", "DLF", "TATACONSUM", "AMBUJACEM", "BANKBARODA",
    "CHOLAFIN", "GAIL", "ICICIPRULI", "ICICIGI", "TVSMOTOR", "BPCL",
    "INDIGO", "ETERNAL", "VBL", "LTM", "SRF", "MARICO", "PNB", "CANBK",
    "PFC", "RECLTD", "TORNTPHARM", "BOSCHLTD", "HAL", "MOTHERSON",
    "PIIND", "BERGEPAINT", "JINDALSTEL", "MUTHOOTFIN", "LUPIN", "UPL",
    "AUROPHARMA", "COLPAL", "ASHOKLEY", "POLYCAB", "MPHASIS", "INDUSTOWER",
    "ALKEM", "GLAND", "BALKRISIND", "ABB", "NAUKRI", "PAGEIND", "OFSS",
    "PETRONET", "GUJGASLTD", "ESCORTS", "VOLTAS", "PERSISTENT", "COFORGE",
    "SAIL", "NMDC", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "AUBANK",
    "BANKINDIA", "UNIONBANK", "INDHOTEL", "OBEROIRLTY", "GODREJPROP",
    "PRESTIGE", "PHOENIXLTD", "LODHA", "MFSL", "SHRIRAMFIN", "LTF",
    "TIINDIA", "SUPREMEIND", "ASTRAL", "CUMMINSIND", "SCHAEFFLER",
    "TIMKEN", "AIAENG", "KEI", "RVNL", "IRFC", "CONCOR", "IRCTC",
    "BHEL", "BDL", "HUDCO", "NHPC", "SJVN", "TATAPOWER", "ADANIPOWER",
    "JSWENERGY", "TORNTPOWER", "CESC", "NLCINDIA", "IEX", "MAZDOCK",
    "COCHINSHIP", "GRSE", "GMRAIRPORT", "IGL", "ATGL", "GUJGASLTD",
    "DEEPAKNTR", "AARTIIND", "TATACHEM", "NAVINFLUOR", "FLUOROCHEM",
    "LAURUSLABS", "GLENMARK", "ZYDUSLIFE", "BIOCON", "IPCALAB",
    "ABBOTINDIA", "SANOFI", "PFIZER", "GLAXO", "SYNGENE", "MANKIND",
    "POLICYBZR", "PAYTM", "NYKAA", "DELHIVERY", "DMART", "TRIDENT",
    "PAGEIND", "RAYMOND", "ABFRL", "KALYANKJIL", "RAJESHEXPO",
    "TITAGARH", "JYOTHYLAB", "EMAMILTD", "GILLETTE", "VGUARD",
    "WHIRLPOOL", "CROMPTON", "AMBER", "DIXON", "KAYNES", "SYRMA",
    "CYIENT", "TATAELXSI", "KPITTECH", "LTTS", "HEXW", "FSL",
    "BSOFT", "NIITLTD", "ZENSARTECH", "SONACOMS", "BHARATFORG",
    "MRF", "APOLLOTYRE", "CEATLTD", "EXIDEIND", "ARE&M",
]
# Deduplicate while preserving order
NIFTY_200_SYMBOLS = list(dict.fromkeys(NIFTY_200_SYMBOLS))


# ============================================================================
# ANGEL ONE AUTH
# ============================================================================

def angel_login() -> SmartConnect:
    """Logs into Angel One SmartAPI using TOTP and returns an authenticated
    SmartConnect session object."""
    if not all([Config.CLIENT_ID, Config.API_KEY, Config.TOTP_SECRET, Config.MPIN]):
        raise RuntimeError(
            "Missing Angel One credentials. Ensure ANGEL_CLIENT_ID, ANGEL_API_KEY, "
            "ANGEL_TOTP_SECRET and ANGEL_MPIN are set as environment variables."
        )

    smart_api = SmartConnect(api_key=Config.API_KEY)
    totp = pyotp.TOTP(Config.TOTP_SECRET).now()

    session = smart_api.generateSession(Config.CLIENT_ID, Config.MPIN, totp)
    if not session or not session.get("status"):
        raise RuntimeError(f"Angel One login failed: {session}")

    log.info("Angel One login successful for client %s", Config.CLIENT_ID)
    return smart_api


# ============================================================================
# INSTRUMENT MASTER (token lookup)
# ============================================================================

INSTRUMENT_MASTER_URL = (
    "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
)


def load_instrument_master(force_refresh: bool = False) -> pd.DataFrame:
    """Returns Angel One's instrument master as a DataFrame, using a local
    CSV cache that's refreshed at most once every
    Config.INSTRUMENT_MASTER_MAX_AGE_HOURS hours (default 24h). This avoids
    re-downloading the (fairly large) instrument master on every run."""
    cache_path = Config.INSTRUMENT_MASTER_CACHE_PATH

    if not force_refresh and os.path.exists(cache_path):
        age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600.0
        if age_hours < Config.INSTRUMENT_MASTER_MAX_AGE_HOURS:
            log.info("Loading instrument master from cache (%.1fh old)", age_hours)
            return pd.read_csv(cache_path, dtype={"token": str})
        else:
            log.info("Instrument master cache is stale (%.1fh old), refreshing", age_hours)

    import urllib.request

    log.info("Downloading Angel One instrument master...")
    with urllib.request.urlopen(INSTRUMENT_MASTER_URL, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    df = pd.DataFrame(data)
    # Equity cash-market symbols carry the '-EQ' suffix in Angel One's master
    df = df[(df["exch_seg"] == "NSE") & (df["symbol"].str.endswith("-EQ"))]
    df["base_symbol"] = df["symbol"].str.replace("-EQ", "", regex=False)
    df = df[["token", "symbol", "base_symbol", "name", "lotsize"]]

    df.to_csv(cache_path, index=False)
    log.info("Instrument master cached to %s (%d rows)", cache_path, len(df))
    return df


def build_symbol_token_map(symbols: list[str]) -> dict[str, str]:
    """Returns {trading_symbol: token} for the given list of base symbols."""
    master = load_instrument_master()
    lookup = master.drop_duplicates("base_symbol").set_index("base_symbol")["token"]
    token_map = {}
    missing = []
    for sym in symbols:
        if sym in lookup.index:
            token_map[sym] = str(lookup.loc[sym])
        else:
            missing.append(sym)
    if missing:
        log.warning("No instrument token found for %d symbols: %s",
                    len(missing), missing[:20])
    return token_map


# ============================================================================
# HISTORICAL CANDLE FETCH
# ============================================================================

def _is_rate_limit_response(resp_or_error: object) -> bool:
    """Checks an API response dict or exception/string for known Angel One
    rate-limit phrasing, so we can react differently (longer sleep) than a
    generic transient error."""
    text = ""
    if isinstance(resp_or_error, dict):
        text = json.dumps(resp_or_error).lower()
    else:
        text = str(resp_or_error).lower()
    return any(marker in text for marker in Config.RATE_LIMIT_MARKERS)


def fetch_historical_candles(
    smart_api: SmartConnect,
    token: str,
    from_dt: dt.datetime,
    to_dt: dt.datetime,
    interval: str = Config.CANDLE_INTERVAL,
) -> pd.DataFrame:
    """Fetches historical candles for a single instrument token between
    from_dt and to_dt, with retry handling for transient HTTP/502 errors AND
    dedicated handling for Angel One's rate-limit responses ("Access denied
    because of exceeding access rate"). Angel One limits each request to
    roughly Config.CHUNK_DAYS days of 5-min data, so this function
    internally chunks the date range and concatenates results."""

    all_rows = []
    chunk_start = from_dt
    chunk_size = dt.timedelta(days=Config.CHUNK_DAYS)

    while chunk_start < to_dt:
        chunk_end = min(chunk_start + chunk_size, to_dt)

        params = {
            "exchange": Config.EXCHANGE,
            "symboltoken": token,
            "interval": interval,
            "fromdate": chunk_start.strftime("%Y-%m-%d %H:%M"),
            "todate": chunk_end.strftime("%Y-%m-%d %H:%M"),
        }

        rows = None
        for attempt in range(1, Config.MAX_RETRIES + 1):
            try:
                resp = smart_api.getCandleData(params)

                if resp and resp.get("status") and resp.get("data"):
                    rows = resp["data"]
                    break

                if _is_rate_limit_response(resp):
                    log.warning(
                        "RATE LIMIT hit for token %s [%s -> %s] (attempt %d). "
                        "Sleeping %ds before retry.",
                        token, chunk_start.date(), chunk_end.date(),
                        attempt, Config.RATE_LIMIT_SLEEP_SEC,
                    )
                    time.sleep(Config.RATE_LIMIT_SLEEP_SEC)
                    continue  # don't count this against the normal backoff curve below

                log.warning(
                    "Empty/failed response for token %s [%s -> %s] (attempt %d): %s",
                    token, chunk_start.date(), chunk_end.date(), attempt, resp,
                )

            except Exception as e:
                if _is_rate_limit_response(e):
                    log.warning(
                        "RATE LIMIT exception for token %s [%s -> %s] (attempt %d): %s. "
                        "Sleeping %ds before retry.",
                        token, chunk_start.date(), chunk_end.date(),
                        attempt, e, Config.RATE_LIMIT_SLEEP_SEC,
                    )
                    time.sleep(Config.RATE_LIMIT_SLEEP_SEC)
                    continue
                log.warning(
                    "Error fetching candles for token %s [%s -> %s] (attempt %d): %s",
                    token, chunk_start.date(), chunk_end.date(), attempt, e,
                )

            # Generic transient-error backoff (linear w/ attempt number)
            time.sleep(Config.RETRY_BACKOFF_SEC * attempt)

        if rows:
            all_rows.extend(rows)
        else:
            log.error(
                "Giving up on chunk [%s -> %s] for token %s after %d attempts",
                chunk_start.date(), chunk_end.date(), token, Config.MAX_RETRIES,
            )

        time.sleep(Config.REQUEST_SLEEP_SEC)
        chunk_start = chunk_end

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    # Angel One returns timestamps with a +05:30 offset (e.g. "...T09:15:00+05:30").
    # We normalize to timezone-naive IST wall-clock time so all downstream
    # comparisons (cache gap detection, market-window checks, etc.) are
    # consistent with the naive datetimes used elsewhere in this script.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False).dt.tz_localize(None)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_all_symbols(
    smart_api: SmartConnect,
    token_map: dict[str, str],
    lookback_days: int = Config.LOOKBACK_DAYS,
) -> dict[str, pd.DataFrame]:
    """Fetches historical 5-min candles for every symbol in token_map (no
    caching -- use load_or_fetch_all_symbols for the incremental-cache
    version used by main())."""
    to_dt = dt.datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
    from_dt = to_dt - dt.timedelta(days=lookback_days)

    data = {}
    for i, (sym, token) in enumerate(token_map.items(), 1):
        log.info("[%d/%d] Fetching %s (token=%s)...", i, len(token_map), sym, token)
        df = fetch_historical_candles(smart_api, token, from_dt, to_dt)
        if df.empty:
            log.warning("No data returned for %s, skipping.", sym)
            continue
        data[sym] = df
    return data


# ============================================================================
# LOCAL CACHE (incremental, CSV-based)
# ============================================================================
# Goal: first run downloads the full lookback window per symbol (slow, many
# requests). Every subsequent run only fetches candles AFTER the last
# cached timestamp, so a daily re-run is fast and uses far fewer API calls.
# Cache files are plain CSV (not pickle) so they're diffable/inspectable and
# trivially portable as a GitHub Actions artifact between runs.

CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def candle_cache_path(symbol: str) -> str:
    return os.path.join(Config.CANDLE_CACHE_DIR, f"{symbol}.csv")


def _read_cached_candles(symbol: str) -> pd.DataFrame:
    path = candle_cache_path(symbol)
    if not os.path.exists(path):
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _write_cached_candles(symbol: str, df: pd.DataFrame) -> None:
    os.makedirs(Config.CANDLE_CACHE_DIR, exist_ok=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df.to_csv(candle_cache_path(symbol), index=False)


def load_or_fetch_all_symbols(
    smart_api: SmartConnect,
    token_map: dict[str, str],
    lookback_days: int = Config.LOOKBACK_DAYS,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """For each symbol:
      - No cache (or force_refresh=True): download the full lookback window
        and write it to CSV.
      - Cache exists: read the last cached timestamp and fetch only candles
        from (last_timestamp + 5 minutes) up to now, then append + dedupe +
        re-save. This is what makes subsequent runs fast (typically a
        handful of candles per symbol instead of months of history).

    Returns {symbol: full_dataframe_including_cache}.
    """
    os.makedirs(Config.CANDLE_CACHE_DIR, exist_ok=True)
    to_dt = dt.datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
    full_from_dt = to_dt - dt.timedelta(days=lookback_days)

    data = {}
    n = len(token_map)
    for i, (sym, token) in enumerate(token_map.items(), 1):
        cached_df = pd.DataFrame(columns=CANDLE_COLUMNS) if force_refresh else _read_cached_candles(sym)

        if cached_df.empty:
            log.info("[%d/%d] %s: no cache found, fetching full %d-day history...",
                      i, n, sym, lookback_days)
            new_df = fetch_historical_candles(smart_api, token, full_from_dt, to_dt)
            if new_df.empty:
                log.warning("No data returned for %s, skipping.", sym)
                continue
            _write_cached_candles(sym, new_df)
            data[sym] = new_df
            continue

        last_ts = cached_df["timestamp"].max()
        gap_start = last_ts + dt.timedelta(minutes=5)

        if gap_start >= to_dt:
            log.info("[%d/%d] %s: cache already up to date (last candle %s), skipping fetch",
                      i, n, sym, last_ts)
            data[sym] = cached_df
            continue

        log.info("[%d/%d] %s: cache found (last candle %s), fetching only the gap -> %s",
                  i, n, sym, last_ts, to_dt)
        new_df = fetch_historical_candles(smart_api, token, gap_start, to_dt)

        if new_df.empty:
            log.info("[%d/%d] %s: no new candles in the gap (market closed / holiday?)", i, n, sym)
            data[sym] = cached_df
            continue

        combined = pd.concat([cached_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        _write_cached_candles(sym, combined)
        log.info("[%d/%d] %s: appended %d new candle(s), cache now has %d rows",
                  i, n, sym, len(new_df), len(combined))
        data[sym] = combined

    return data


# ============================================================================
# INDICATORS: VWAP (daily reset) and PDH/PDL
# ============================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 'date', 'vwap' (resets every trading day), 'pdh', 'pdl'
    (previous day's high/low) and 'ema5' columns to a 5-min candle
    DataFrame. Expects columns: timestamp, open, high, low, close, volume."""
    df = df.copy()
    df["date"] = df["timestamp"].dt.date

    # --- VWAP, reset each trading day ---
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    df["tp_vol"] = typical_price * df["volume"]
    df["cum_tp_vol"] = df.groupby("date")["tp_vol"].cumsum()
    df["cum_vol"] = df.groupby("date")["volume"].cumsum()
    df["vwap"] = df["cum_tp_vol"] / df["cum_vol"].replace(0, np.nan)
    df.drop(columns=["tp_vol", "cum_tp_vol", "cum_vol"], inplace=True)

    # --- EMA5 (kept for reference / future use, not used in entry trigger) ---
    df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()

    # --- Previous Day High / Low ---
    daily = df.groupby("date").agg(day_high=("high", "max"), day_low=("low", "min"))
    daily["pdh"] = daily["day_high"].shift(1)
    daily["pdl"] = daily["day_low"].shift(1)
    df = df.merge(daily[["pdh", "pdl"]], left_on="date", right_index=True, how="left")

    return df


# ============================================================================
# TRADE RECORD
# ============================================================================

@dataclass
class Trade:
    symbol: str
    side: str                  # "LONG" or "SHORT"
    setup_date: dt.date
    rejection_time: dt.datetime
    trigger_time: dt.datetime
    entry_time: dt.datetime = None
    entry_price: float = None
    sl_price: float = None
    target_price: float = None
    qty: int = 0
    exit_time: dt.datetime = None
    exit_price: float = None
    exit_reason: str = None    # "SL", "TARGET1_TRAIL_EXIT", "EOD"
    hit_target1: bool = False
    pnl: float = None
    rr_achieved: float = None
    planned_rr: float = None


# ============================================================================
# CORE STRATEGY / BACKTEST ENGINE (per stock)
# ============================================================================

def find_swing_low(candles: pd.DataFrame, idx: int, lookback: int = Config.SWING_LOOKBACK) -> float:
    """Swing low = lowest low among the last `lookback` candles up to and
    including idx."""
    start = max(0, idx - lookback + 1)
    return candles["low"].iloc[start: idx + 1].min()


def find_swing_high(candles: pd.DataFrame, idx: int, lookback: int = Config.SWING_LOOKBACK) -> float:
    """Swing high = highest high among the last `lookback` candles up to and
    including idx."""
    start = max(0, idx - lookback + 1)
    return candles["high"].iloc[start: idx + 1].max()


def backtest_symbol(symbol: str, df: pd.DataFrame) -> list[Trade]:
    """Runs the full PDH/PDL rejection strategy on a single symbol's 5-min
    candle history (with indicators already attached) and returns the list
    of completed Trade objects.

    State machine per trading day:
      WAITING_FOR_REJECTION
        -> on rejection candle (wick touches PDL/PDH, closes back inside):
           remember rejection level and side, move to WAITING_FOR_VWAP_CROSS
      WAITING_FOR_VWAP_CROSS (must be a LATER candle than the rejection one)
        -> on candle crossing VWAP in the trigger direction and closing
           beyond it: compute entry/SL/target, place a pending limit entry
           at next opportunity -> move to PENDING_ENTRY
      PENDING_ENTRY
        -> entry fills when price trades through the limit price (we
           approximate fills using subsequent candle highs/lows)
      IN_POSITION
        -> manage SL / Target1 / trailing-after-target1 until exit
    Only one trade per symbol per day is allowed; once a rejection setup is
    used (or a later one would form) for the day, the day is otherwise
    considered "done" once a trade has actually been entered.
    """
    trades: list[Trade] = []
    if df.empty or df["pdh"].isna().all():
        return trades

    df = df.reset_index(drop=True)
    dates = sorted(df["date"].unique())

    for day in dates:
        day_mask = df["date"] == day
        day_idx = df.index[day_mask]
        if len(day_idx) == 0:
            continue

        pdh = df.loc[day_idx[0], "pdh"]
        pdl = df.loc[day_idx[0], "pdl"]
        if pd.isna(pdh) or pd.isna(pdl):
            continue  # first day in dataset, no previous day reference

        traded_today = False

        # Track separately for long (PDL) and short (PDH) setups since both
        # sides can be "armed" simultaneously, but only one trade total/day.
        long_rejection_idx = None
        short_rejection_idx = None

        for i in day_idx:
            candle_time = df.loc[i, "timestamp"]
            candle_t = candle_time.time()

            if traded_today:
                break  # max 1 trade/stock/day

            # Only look for NEW rejection/trigger setups inside the entry window.
            # (A position, once entered, can still be managed after 10:30 —
            # that is handled in the position-management loop below.)
            if candle_t > Config.ENTRY_WINDOW_END:
                break

            row = df.loc[i]

            # --- Step 1: detect rejection candles (can occur any time before window end) ---
            if long_rejection_idx is None:
                if row["low"] <= pdl and row["close"] > pdl:
                    long_rejection_idx = i

            if short_rejection_idx is None:
                if row["high"] >= pdh and row["close"] < pdh:
                    short_rejection_idx = i

            # --- Step 2: look for VWAP-cross trigger on a LATER candle ---
            if candle_t < Config.ENTRY_WINDOW_START:
                continue  # don't trigger entries before market truly opens or pre-9:15 noise

            trade = None

            if long_rejection_idx is not None and i > long_rejection_idx:
                prev_row = df.loc[i - 1] if i > 0 else None
                crossed_up = (
                    prev_row is not None
                    and prev_row["close"] <= prev_row["vwap"]
                    and row["close"] > row["vwap"]
                )
                if crossed_up:
                    trade = build_long_trade(symbol, df, i, long_rejection_idx, pdl)

            if trade is None and short_rejection_idx is not None and i > short_rejection_idx:
                prev_row = df.loc[i - 1] if i > 0 else None
                crossed_down = (
                    prev_row is not None
                    and prev_row["close"] >= prev_row["vwap"]
                    and row["close"] < row["vwap"]
                )
                if crossed_down:
                    trade = build_short_trade(symbol, df, i, short_rejection_idx, pdh)

            if trade is not None:
                filled_trade = simulate_entry_and_exit(df, i, trade)
                if filled_trade is not None:
                    trades.append(filled_trade)
                    traded_today = True

    return trades


def build_long_trade(symbol: str, df: pd.DataFrame, trigger_idx: int,
                      rejection_idx: int, pdl: float) -> Trade | None:
    """Constructs a planned LONG trade off a VWAP-cross trigger candle.
    Returns None if the RR filter rejects the trade."""
    trigger_row = df.loc[trigger_idx]
    entry_price = trigger_row["close"] + Config.ENTRY_BUFFER
    sl_price = pdl

    risk = entry_price - sl_price
    if risk <= 0:
        return None  # malformed setup (PDL above entry?), skip

    target_price = entry_price + Config.TARGET_RR * risk
    planned_rr = Config.TARGET_RR  # by construction; min-RR filter below double-checks

    if (target_price - entry_price) / risk < Config.MIN_RR:
        return None  # RR filter: skip trade entirely

    return Trade(
        symbol=symbol,
        side="LONG",
        setup_date=trigger_row["date"],
        rejection_time=df.loc[rejection_idx, "timestamp"],
        trigger_time=trigger_row["timestamp"],
        entry_price=round(entry_price, 2),
        sl_price=round(sl_price, 2),
        target_price=round(target_price, 2),
        planned_rr=planned_rr,
    )


def build_short_trade(symbol: str, df: pd.DataFrame, trigger_idx: int,
                       rejection_idx: int, pdh: float) -> Trade | None:
    """Constructs a planned SHORT trade off a VWAP-cross trigger candle.
    Returns None if the RR filter rejects the trade."""
    trigger_row = df.loc[trigger_idx]
    entry_price = trigger_row["close"] - Config.ENTRY_BUFFER
    sl_price = pdh

    risk = sl_price - entry_price
    if risk <= 0:
        return None  # malformed setup (PDH below entry?), skip

    target_price = entry_price - Config.TARGET_RR * risk
    planned_rr = Config.TARGET_RR

    if (entry_price - target_price) / risk < Config.MIN_RR:
        return None

    return Trade(
        symbol=symbol,
        side="SHORT",
        setup_date=trigger_row["date"],
        rejection_time=df.loc[rejection_idx, "timestamp"],
        trigger_time=trigger_row["timestamp"],
        entry_price=round(entry_price, 2),
        sl_price=round(sl_price, 2),
        target_price=round(target_price, 2),
        planned_rr=planned_rr,
    )


def simulate_entry_and_exit(df: pd.DataFrame, trigger_idx: int, trade: Trade) -> Trade | None:
    """Simulates whether the pending limit entry fills, and if so, manages
    the position candle-by-candle: SL, Target1, then trailing-on-swing-break
    after Target1 is hit. Returns the completed Trade, or None if it never
    fills (e.g. price runs away before reaching the limit price) or there's
    no more data for the day."""
    day = trade.setup_date
    is_long = trade.side == "LONG"

    # --- Search for entry fill starting from the candle AFTER the trigger candle ---
    entry_idx = None
    for j in range(trigger_idx + 1, len(df)):
        row = df.loc[j]
        if row["date"] != day:
            return None  # day ended without a fill
        if is_long and row["high"] >= trade.entry_price:
            entry_idx = j
            break
        if (not is_long) and row["low"] <= trade.entry_price:
            entry_idx = j
            break

    if entry_idx is None:
        return None  # never filled

    trade.entry_time = df.loc[entry_idx, "timestamp"]

    # Fixed-risk position sizing
    risk_per_share = abs(trade.entry_price - trade.sl_price)
    trade.qty = max(1, math.floor(Config.RISK_PER_TRADE / risk_per_share)) if risk_per_share > 0 else 0
    if trade.qty <= 0:
        return None

    # --- Manage the open position candle by candle ---
    trailing_sl = trade.sl_price
    hit_target1 = False

    for k in range(entry_idx, len(df)):
        row = df.loc[k]
        if row["date"] != day:
            # Position carried beyond available data for this run -> close at last close (EOD safeguard)
            trade.exit_time = df.loc[k - 1, "timestamp"]
            trade.exit_price = df.loc[k - 1, "close"]
            trade.exit_reason = "EOD"
            break

        if is_long:
            # --- Before Target1: fixed SL ---
            if not hit_target1:
                if row["low"] <= trailing_sl:
                    trade.exit_time = row["timestamp"]
                    trade.exit_price = trailing_sl
                    trade.exit_reason = "SL"
                    break
                if row["high"] >= trade.target_price:
                    hit_target1 = True
                    trade.hit_target1 = True
                    # initialize trailing reference at the most recent confirmed swing low
                    trailing_sl = find_swing_low(df, k)
                    continue
            else:
                # --- After Target1: trail using swing low, update only on new swing ---
                new_swing_low = find_swing_low(df, k)
                if new_swing_low > trailing_sl:
                    trailing_sl = new_swing_low
                if row["close"] < trailing_sl:
                    trade.exit_time = row["timestamp"]
                    trade.exit_price = row["close"]
                    trade.exit_reason = "TARGET1_TRAIL_EXIT"
                    break

        else:  # SHORT
            if not hit_target1:
                if row["high"] >= trailing_sl:
                    trade.exit_time = row["timestamp"]
                    trade.exit_price = trailing_sl
                    trade.exit_reason = "SL"
                    break
                if row["low"] <= trade.target_price:
                    hit_target1 = True
                    trade.hit_target1 = True
                    trailing_sl = find_swing_high(df, k)
                    continue
            else:
                new_swing_high = find_swing_high(df, k)
                if new_swing_high < trailing_sl:
                    trailing_sl = new_swing_high
                if row["close"] > trailing_sl:
                    trade.exit_time = row["timestamp"]
                    trade.exit_price = row["close"]
                    trade.exit_reason = "TARGET1_TRAIL_EXIT"
                    break

        # Hard EOD exit at/after market close on the same day
        if row["timestamp"].time() >= Config.MARKET_CLOSE:
            trade.exit_time = row["timestamp"]
            trade.exit_price = row["close"]
            trade.exit_reason = "EOD"
            break

    if trade.exit_time is None:
        # Ran out of candles entirely (end of dataset) without a clean exit
        last_row = df.iloc[-1]
        trade.exit_time = last_row["timestamp"]
        trade.exit_price = last_row["close"]
        trade.exit_reason = "EOD_DATA_END"

    # --- P&L and realised RR ---
    if is_long:
        trade.pnl = (trade.exit_price - trade.entry_price) * trade.qty
    else:
        trade.pnl = (trade.entry_price - trade.exit_price) * trade.qty

    if risk_per_share > 0:
        signed_move = (trade.exit_price - trade.entry_price) if is_long else (trade.entry_price - trade.exit_price)
        trade.rr_achieved = round(signed_move / risk_per_share, 2)

    return trade


# ============================================================================
# PORTFOLIO-LEVEL CONSTRAINT: MAX CONCURRENT POSITIONS
# ============================================================================

def apply_max_concurrent_positions(all_trades: list[Trade],
                                    max_concurrent: int = Config.MAX_CONCURRENT_POSITIONS) -> list[Trade]:
    """Given trades generated independently per symbol (which may overlap in
    time), this enforces a portfolio-wide cap on simultaneously open
    positions. Trades are considered chronologically by entry_time; any
    trade whose entry_time would exceed the concurrency cap (given other
    already-accepted open trades at that moment) is dropped entirely --
    mirroring a live system that simply wouldn't take the signal because
    its allotted "slots" were full.

    Note: this is a simplifying assumption (first-come-first-served by
    entry time). It does not try to be "smart" about which trade to prefer.
    """
    candidates = sorted(
        [t for t in all_trades if t.entry_time is not None],
        key=lambda t: t.entry_time,
    )

    accepted: list[Trade] = []
    open_positions: list[Trade] = []  # trades currently considered "open"

    for t in candidates:
        # Drop from open_positions any trade that has already exited by t.entry_time
        open_positions = [p for p in open_positions if p.exit_time > t.entry_time]

        if len(open_positions) < max_concurrent:
            accepted.append(t)
            open_positions.append(t)
        # else: signal dropped, portfolio was full

    return accepted


# ============================================================================
# REPORTING
# ============================================================================

def summarize_trades(trades: list[Trade]) -> None:
    if not trades:
        log.info("No trades generated by the strategy in this backtest window.")
        return

    df = pd.DataFrame([t.__dict__ for t in trades])
    df.to_csv("trade_log.csv", index=False)

    total_trades = len(df)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    win_rate = 100 * len(wins) / total_trades if total_trades else 0
    total_pnl = df["pnl"].sum()
    avg_rr_achieved = df["rr_achieved"].mean()
    avg_win = wins["pnl"].mean() if len(wins) else 0
    avg_loss = losses["pnl"].mean() if len(losses) else 0
    target1_hit_rate = 100 * df["hit_target1"].sum() / total_trades if total_trades else 0

    print("\n" + "=" * 60)
    print("PDH/PDL REJECTION STRATEGY — BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Total trades taken      : {total_trades}")
    print(f"Winning trades          : {len(wins)}")
    print(f"Losing trades           : {len(losses)}")
    print(f"Win rate                : {win_rate:.2f}%")
    print(f"Target1 hit rate        : {target1_hit_rate:.2f}%")
    print(f"Total P&L (Rs)          : {total_pnl:,.2f}")
    print(f"Avg RR achieved         : {avg_rr_achieved:.2f}")
    print(f"Avg win (Rs)            : {avg_win:,.2f}")
    print(f"Avg loss (Rs)           : {avg_loss:,.2f}")
    print("=" * 60)
    print("Full trade log saved to trade_log.csv")
    print("=" * 60 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    log.info("Starting PDH/PDL rejection strategy backtest")
    log.info("Universe size: %d symbols | Lookback: %d days",
              len(NIFTY_200_SYMBOLS), Config.LOOKBACK_DAYS)

    smart_api = angel_login()
    token_map = build_symbol_token_map(NIFTY_200_SYMBOLS)
    log.info("Resolved tokens for %d/%d symbols", len(token_map), len(NIFTY_200_SYMBOLS))

    raw_data = load_or_fetch_all_symbols(smart_api, token_map)

    all_trades: list[Trade] = []
    for symbol, df in raw_data.items():
        try:
            df_ind = add_indicators(df)
            trades = backtest_symbol(symbol, df_ind)
            all_trades.extend(trades)
            log.info("%s: %d trade(s) generated", symbol, len(trades))
        except Exception as e:
            log.error("Error backtesting %s: %s", symbol, e)

    log.info("Total raw signals across universe (pre-concurrency-filter): %d", len(all_trades))

    final_trades = apply_max_concurrent_positions(all_trades)
    log.info("Total trades after max-concurrent-positions filter: %d", len(final_trades))

    summarize_trades(final_trades)


if __name__ == "__main__":
    main()
