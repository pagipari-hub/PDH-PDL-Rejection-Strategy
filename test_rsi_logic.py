"""
Test for daily RSI(14) calculation and the previous-day attachment logic
used to avoid lookahead bias (a trading day's 5-min candles should only see
RSI computed from data up to and including the PRIOR day's close).
"""
import datetime as dt
import pandas as pd
import numpy as np

from pdh_pdl_backtest import compute_daily_rsi, attach_prev_day_rsi, Config


def make_daily_df(closes: list[float], start_date: dt.date) -> pd.DataFrame:
    rows = []
    d = start_date
    for c in closes:
        rows.append({
            "timestamp": pd.Timestamp(d),
            "open": c, "high": c, "low": c, "close": c, "volume": 1000,
        })
        d += dt.timedelta(days=1)
    return pd.DataFrame(rows)


def test_rsi_known_values():
    """Cross-check our Wilder's RSI implementation against a textbook
    example with known inputs. Using a classic monotonic-up-then-down
    series and verifying RSI behaves sensibly (100 in a pure uptrend with
    period+1 candles, since avg_loss=0)."""
    # 16 strictly increasing closes -> after warmup, RSI should be 100
    # (all gains, zero losses observed in the smoothing window)
    closes = [100 + i for i in range(20)]
    daily_df = make_daily_df(closes, dt.date(2026, 1, 1))
    rsi_df = compute_daily_rsi(daily_df, period=14)

    last_rsi = rsi_df["rsi"].iloc[-1]
    print(f"Pure uptrend RSI (last value): {last_rsi}")
    assert last_rsi == 100, f"Expected RSI=100 in pure uptrend, got {last_rsi}"

    # 20 strictly decreasing closes -> RSI should approach 0
    closes_down = [100 - i for i in range(20)]
    daily_df_down = make_daily_df(closes_down, dt.date(2026, 1, 1))
    rsi_df_down = compute_daily_rsi(daily_df_down, period=14)
    last_rsi_down = rsi_df_down["rsi"].iloc[-1]
    print(f"Pure downtrend RSI (last value): {last_rsi_down}")
    assert last_rsi_down == 0, f"Expected RSI=0 in pure downtrend, got {last_rsi_down}"

    print("RSI KNOWN VALUES TEST PASSED ✅")


def test_rsi_warmup_is_nan_before_period():
    """Before `period` candles are available, RSI should be NaN (Wilder's
    method needs a full warmup window)."""
    closes = [100 + (i % 3) for i in range(10)]  # only 10 candles, period=14
    daily_df = make_daily_df(closes, dt.date(2026, 1, 1))
    rsi_df = compute_daily_rsi(daily_df, period=14)
    assert rsi_df["rsi"].isna().all(), "Expected all-NaN RSI with insufficient warmup data"
    print("RSI WARMUP NaN TEST PASSED ✅")


def test_prev_day_attachment_no_lookahead():
    """The core correctness property: every 5-min candle on day D should
    see the RSI computed using day D-1's close, NEVER day D's own close
    (which wouldn't exist yet at 9:15 AM on day D)."""
    # Build daily RSI series: day1=NaN(warmup), day2=50.0, day3=70.0, day4=30.0
    daily_rsi_df = pd.DataFrame({
        "date": [dt.date(2026, 1, 5), dt.date(2026, 1, 6), dt.date(2026, 1, 7), dt.date(2026, 1, 8)],
        "rsi": [np.nan, 50.0, 70.0, 30.0],
    })

    # Build intraday candles for day3 (2026-01-07) and day4 (2026-01-08)
    intraday_rows = []
    for t in ["09:15", "09:20", "09:25"]:
        ts = pd.Timestamp(f"2026-01-07 {t}")
        intraday_rows.append({"timestamp": ts, "date": dt.date(2026, 1, 7), "close": 100})
    for t in ["09:15", "09:20"]:
        ts = pd.Timestamp(f"2026-01-08 {t}")
        intraday_rows.append({"timestamp": ts, "date": dt.date(2026, 1, 8), "close": 100})
    intraday_df = pd.DataFrame(intraday_rows)

    result = attach_prev_day_rsi(intraday_df, daily_rsi_df)
    print(result[["timestamp", "date", "rsi"]].to_string(index=False))

    # Day3 (2026-01-07) candles should see day2's RSI (50.0), NOT day3's own (70.0)
    day3_rsi = result[result["date"] == dt.date(2026, 1, 7)]["rsi"].unique()
    assert len(day3_rsi) == 1 and day3_rsi[0] == 50.0, f"Expected day3 candles to see RSI=50.0 (day2's), got {day3_rsi}"

    # Day4 (2026-01-08) candles should see day3's RSI (70.0), NOT day4's own (30.0)
    day4_rsi = result[result["date"] == dt.date(2026, 1, 8)]["rsi"].unique()
    assert len(day4_rsi) == 1 and day4_rsi[0] == 70.0, f"Expected day4 candles to see RSI=70.0 (day3's), got {day4_rsi}"

    print("PREV-DAY ATTACHMENT (NO LOOKAHEAD) TEST PASSED ✅")


if __name__ == "__main__":
    test_rsi_known_values()
    test_rsi_warmup_is_nan_before_period()
    test_prev_day_attachment_no_lookahead()
    print("\n🎉 ALL RSI TESTS PASSED")
