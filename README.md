# PDH/PDL Rejection Strategy — Backtest

Intraday strategy backtest for Nifty 200 stocks using Angel One SmartAPI
historical 5-min candle data.

## Strategy Rules

- **Universe:** Nifty 200 stocks (5-min candles)
- **Entry window:** 09:15 – 10:30 IST only
- **Setup (LONG @ PDL):**
  1. Rejection candle: wick touches/pierces PDL, candle closes back above PDL
  2. A later candle crosses VWAP upward and closes above it (trigger)
  3. Entry = trigger candle's close + ₹0.10 (limit order)
  4. SL = PDL
  5. Skip the trade if achievable RR < 1:1.5
  6. Target 1 = 1:2 RR
  7. After Target 1: trail using the last confirmed 5-candle swing low
     (fixed reference until a new swing low forms); exit on a close below it
- **Setup (SHORT @ PDH):** mirror of the above
- **Constraints:** max 1 trade/stock/day, max 3 concurrent open positions
  across the portfolio
- **Position sizing:** fixed risk per trade (₹1000 default), quantity =
  risk ÷ |entry − SL|

### Daily RSI (analysis / not yet an active filter)

Each trade also records the **previous trading day's** daily RSI(14) at
the rejection candle's date — `RSI > 60` for PDH (short) rejections,
`RSI < 40` for PDL (long) rejections is the hypothesis being tested
(overbought/oversold confirmation for the rejection). Using the *previous*
day's RSI (not the current, still-forming day) avoids lookahead bias: at
9:15 AM the current day's daily candle hasn't closed yet.

This is currently recorded for analysis (visible as the `rsi` column in
`trade_log.csv`, plus a `RSI condition MET` / `NOT met` breakdown printed
in the run summary) but is **not yet wired in as a hard entry filter** —
that's a deliberate next step once the backtest results confirm whether it
actually improves performance.

## Setup

1. Add these as **GitHub repo secrets** (Settings → Secrets and variables →
   Actions → New repository secret):
   - `ANGEL_CLIENT_ID`
   - `ANGEL_API_KEY`
   - `ANGEL_TOTP_SECRET`
   - `ANGEL_MPIN`
2. Go to the **Actions** tab → "PDH-PDL Backtest" workflow → **Run workflow**
   (manual trigger).
3. Once it finishes, download the `trade-log` artifact for the full trade
   list (`trade_log.csv`), and check the run logs for the summary (win
   rate, total P&L, avg RR achieved).

## Performance characteristics (incremental cache)

- **First run**: no cache exists yet, so every symbol's full
  `LOOKBACK_DAYS` history is downloaded in `CHUNK_DAYS`-sized requests
  (~6 requests/symbol at 180 days / 30-day chunks, ~206 symbols ≈ 1,200
  requests). This can take 1-2 hours depending on Angel One's rate limits.
- **Every subsequent run**: each symbol's CSV cache is read, and only
  candles after the last cached timestamp are fetched (typically 1 request
  per symbol, often covering well under a day of gap). Expect ~200 requests
  total and a 2-10 minute runtime.
- The cache is persisted between runs via the **GitHub Actions cache**
  (`actions/cache`), since GitHub-hosted runners are ephemeral and don't
  retain the filesystem between jobs. See the "GitHub Actions cache
  details" section below if you want to inspect or reset it.

## Local / Colab run

```bash
pip install -r requirements.txt
export ANGEL_CLIENT_ID=...
export ANGEL_API_KEY=...
export ANGEL_TOTP_SECRET=...
export ANGEL_MPIN=...
python pdh_pdl_backtest.py
```

Locally, the cache just lives on disk in `candle_cache/`,
`daily_candle_cache/` and `instrument_master.csv` — no special setup
needed, it'll incrementally update on every run automatically.

## Files

- `pdh_pdl_backtest.py` — main backtest engine (auth, incremental data
  fetch/cache for both 5-min and daily candles, indicators including daily
  RSI, strategy state machine, portfolio concurrency filter, reporting)
- `test_strategy_logic.py` — synthetic-data unit tests validating the
  rejection → VWAP-cross → entry → SL/Target1/trail logic for both long and
  short setups
- `test_data_layer.py` — unit tests for the incremental CSV cache, rate-limit
  detection/retry behavior, and instrument master caching, using a mocked
  Angel One API (no real network access needed)
- `test_rsi_logic.py` — unit tests for the daily RSI(14) calculation
  (cross-checked against known pure-uptrend/downtrend cases) and the
  previous-day attachment logic (verifies no lookahead bias)
- `requirements.txt` — Python dependencies
- `.github/workflows/backtest.yml` — GitHub Actions workflow (manual
  trigger; scheduled run commented out, uncomment if you want it automatic).
  Includes `actions/cache` steps to persist both candle caches between runs.
- `.gitignore` — keeps cache files and trade logs out of git (they're
  persisted via the GitHub Actions cache instead, not committed)

## Rate-limit handling

Angel One's historical API returns an "Access denied because of exceeding
access rate" message when you request too fast. The script specifically
detects this phrase (and a few related ones) and reacts differently than a
generic transient error:

- **Rate-limit detected**: sleeps `RATE_LIMIT_SLEEP_SEC` (45s default) and
  retries, without burning down the normal retry-attempt budget as
  aggressively.
- **Generic transient error** (e.g. a stray 502): linear backoff
  (`RETRY_BACKOFF_SEC * attempt`), up to `MAX_RETRIES` (8 default).
- A small fixed `REQUEST_SLEEP_SEC` (1.0s) pause happens between every
  request regardless of outcome, to stay generally polite to the API.

These are all tunable in the `Config` class if you find Angel One's actual
limits are stricter or looser than assumed.

## Notes / things to verify when you run it for real

- **Nifty 200 symbol list** is a static snapshot baked into the script
  (`NIFTY_200_SYMBOLS`). The index rebalances semi-annually (March/September)
  — update the list periodically.
- **Instrument token lookup** uses Angel One's public instrument master
  JSON, cached locally as `instrument_master.csv` and refreshed once every
  24 hours (`INSTRUMENT_MASTER_MAX_AGE_HOURS`). If a symbol's name changed
  or it's been delisted/renamed, it'll show up in the "no instrument token
  found" warning and be skipped.
- **Candle cache format**: plain CSV per symbol under `candle_cache/`
  (`candle_cache/RELIANCE.csv`, etc.) — human-readable/diffable, unlike the
  old pickle format. Delete the folder (or pass `force_refresh=True` to
  `load_or_fetch_all_symbols`) to force a clean full re-fetch.
- **GitHub Actions cache details**: the workflow saves the cache under a
  key suffixed with the run ID (so it always saves fresh data at the end of
  every run) and restores from the most recent previous run via a
  prefix-matched `restore-keys`. GitHub auto-evicts caches after ~7 days of
  no access, or once the repo's overall 10GB cache quota is hit — so if a
  workflow ever goes a week without running, expect the next run to fall
  back to a full re-fetch.
- **Concurrency filter is first-come-first-served** by entry time — it
  doesn't try to pick the "best" trade when more than 3 signals are
  simultaneously active; it just drops whichever ones overflow the cap, in
  chronological order. Worth knowing if win rate looks different than
  expected.
