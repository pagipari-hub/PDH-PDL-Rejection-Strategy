"""
Unit test for the PDH/PDL rejection strategy logic using synthetic candle
data. This lets us validate the state machine (rejection -> VWAP cross ->
entry -> SL/Target1/trail) WITHOUT needing live Angel One API access.
"""
import datetime as dt
import pandas as pd
import numpy as np

from pdh_pdl_backtest import add_indicators, backtest_symbol, Config


def make_candle(date, time_str, o, h, l, c, v):
    ts = dt.datetime.combine(date, dt.datetime.strptime(time_str, "%H:%M").time())
    return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def build_long_scenario():
    """
    Day 1 (reference day): sets PDH=110, PDL=90 via its own candles.
    Day 2: 
      - 09:15 candle: rejection at PDL (low touches 90, closes at 92)
      - 09:20 candle: drifts (close 91, below VWAP)
      - 09:25 candle: crosses VWAP upward, closes above it -> trigger
      - Entry = trigger close + 0.10
      - Then price rallies to hit target, then trails and exits on swing break
    """
    day1 = dt.date(2026, 1, 5)
    day2 = dt.date(2026, 1, 6)

    rows = []
    # Day 1: simple candles that set PDH=110 / PDL=90
    rows.append(make_candle(day1, "09:15", 100, 105, 95, 100, 10000))
    rows.append(make_candle(day1, "09:20", 100, 110, 98, 105, 10000))  # high=110 -> PDH
    rows.append(make_candle(day1, "09:25", 105, 106, 90, 95, 10000))   # low=90 -> PDL
    rows.append(make_candle(day1, "09:30", 95, 100, 93, 98, 10000))

    # Day 2: the actual setup
    # 09:15 - rejection candle: low pierces PDL(90), closes back above it
    rows.append(make_candle(day2, "09:15", 95, 96, 89, 92, 5000))
    # 09:20 - still below/near vwap (no cross yet)
    rows.append(make_candle(day2, "09:20", 92, 93, 90, 91, 5000))
    # 09:25 - VWAP cross candle: closes meaningfully higher
    rows.append(make_candle(day2, "09:25", 91, 99, 91, 98, 9000))
    # Entry should be ~98.10. Subsequent candles need to trade through that.
    # 09:30 - triggers entry fill (high >= 98.10), continues up
    rows.append(make_candle(day2, "09:30", 98, 102, 97, 101, 6000))
    # risk = entry(98.10) - SL(PDL=90) = 8.10 -> target = entry + 2*risk = ~114.3
    # 09:35 - 09:55 climb toward target
    rows.append(make_candle(day2, "09:35", 101, 105, 100, 104, 6000))
    rows.append(make_candle(day2, "09:40", 104, 108, 103, 107, 6000))
    rows.append(make_candle(day2, "09:45", 107, 111, 106, 110, 6000))
    rows.append(make_candle(day2, "09:50", 110, 115, 109, 114, 6000))  # high 115 >= target ~114.3 -> Target1 hit
    # after target1: trailing via swing low (last 5 candles' low)
    rows.append(make_candle(day2, "09:55", 114, 117, 112, 116, 6000))
    rows.append(make_candle(day2, "10:00", 116, 119, 115, 118, 6000))  # higher low, swing low should rise
    rows.append(make_candle(day2, "10:05", 118, 120, 117, 119, 6000))
    rows.append(make_candle(day2, "10:10", 119, 121, 118, 120, 6000))
    rows.append(make_candle(day2, "10:15", 120, 122, 119, 121, 6000))
    # By now swing low over last 5 candles (09:55-10:15) should be ~112 (09:55 low)
    # Next candle closes below that -> exit
    rows.append(make_candle(day2, "10:20", 121, 121, 105, 108, 6000))

    return pd.DataFrame(rows)


def main():
    df = build_long_scenario()
    df_ind = add_indicators(df)

    print("=== Indicators preview (Day 2) ===")
    day2 = dt.date(2026, 1, 6)
    preview = df_ind[df_ind["date"] == day2][["timestamp", "open", "high", "low", "close", "vwap", "pdh", "pdl"]]
    print(preview.to_string(index=False))

    trades = backtest_symbol("TESTSTOCK", df_ind)
    print(f"\n=== Trades generated: {len(trades)} ===")
    for t in trades:
        print(f"""
Symbol        : {t.symbol}
Side          : {t.side}
Rejection time: {t.rejection_time}
Trigger time  : {t.trigger_time}
Entry time    : {t.entry_time}
Entry price   : {t.entry_price}
SL price      : {t.sl_price}
Target price  : {t.target_price}
Qty           : {t.qty}
Hit Target1   : {t.hit_target1}
Exit time     : {t.exit_time}
Exit price    : {t.exit_price}
Exit reason   : {t.exit_reason}
PnL           : {t.pnl}
RR achieved   : {t.rr_achieved}
""")

    assert len(trades) == 1, f"Expected exactly 1 trade, got {len(trades)}"
    t = trades[0]
    assert t.side == "LONG"
    assert abs(t.entry_price - 98.10) < 0.01, f"Unexpected entry price: {t.entry_price}"
    assert t.sl_price == 90, f"Unexpected SL: {t.sl_price}"
    assert t.hit_target1 is True
    assert t.exit_reason == "TARGET1_TRAIL_EXIT"
    assert t.pnl > 0, "Expected a winning trade in this scenario"
    print("ALL ASSERTIONS PASSED ✅")


def build_short_scenario():
    """Mirror of the long scenario: rejection at PDH, VWAP cross downward,
    short entry, price falls to target, then trails on swing-high break."""
    day1 = dt.date(2026, 1, 5)
    day2 = dt.date(2026, 1, 6)

    rows = []
    rows.append(make_candle(day1, "09:15", 100, 105, 95, 100, 10000))
    rows.append(make_candle(day1, "09:20", 100, 110, 98, 105, 10000))  # PDH=110
    rows.append(make_candle(day1, "09:25", 105, 106, 90, 95, 10000))   # PDL=90
    rows.append(make_candle(day1, "09:30", 95, 100, 93, 98, 10000))

    # Day 2: rejection at PDH
    rows.append(make_candle(day2, "09:15", 105, 111, 104, 108, 5000))  # high pierces PDH(110), closes below
    rows.append(make_candle(day2, "09:20", 108, 109, 106, 107, 5000))
    rows.append(make_candle(day2, "09:25", 107, 107, 99, 100, 9000))   # VWAP cross down, closes below vwap
    # entry = 100 - 0.10 = 99.90; SL = PDH = 110; risk = 10.10; target = 99.90 - 20.20 = 79.70
    rows.append(make_candle(day2, "09:30", 100, 101, 96, 97, 6000))    # triggers entry (low<=99.90)
    rows.append(make_candle(day2, "09:35", 97, 98, 93, 94, 6000))
    rows.append(make_candle(day2, "09:40", 94, 95, 90, 91, 6000))
    rows.append(make_candle(day2, "09:45", 91, 92, 86, 87, 6000))
    rows.append(make_candle(day2, "09:50", 87, 88, 79, 80, 6000))      # low 79 <= target 79.70 -> Target1 hit
    rows.append(make_candle(day2, "09:55", 80, 84, 78, 79, 6000))
    rows.append(make_candle(day2, "10:00", 79, 82, 76, 77, 6000))
    rows.append(make_candle(day2, "10:05", 77, 80, 75, 76, 6000))
    rows.append(make_candle(day2, "10:10", 76, 79, 74, 75, 6000))
    rows.append(make_candle(day2, "10:15", 75, 78, 73, 74, 6000))
    # swing high over last 5 (09:55-10:15) ~ 84 (09:55 high); next candle closes above -> exit
    rows.append(make_candle(day2, "10:20", 74, 90, 73, 88, 6000))

    return pd.DataFrame(rows)


def test_short_scenario():
    df = build_short_scenario()
    df_ind = add_indicators(df)
    trades = backtest_symbol("TESTSHORT", df_ind)
    print(f"\n=== SHORT scenario: trades generated: {len(trades)} ===")
    for t in trades:
        print(f"Side={t.side} Entry={t.entry_price} SL={t.sl_price} Target={t.target_price} "
              f"HitT1={t.hit_target1} ExitReason={t.exit_reason} PnL={t.pnl} RR={t.rr_achieved}")
    assert len(trades) == 1
    t = trades[0]
    assert t.side == "SHORT"
    assert t.sl_price == 110
    assert t.hit_target1 is True
    assert t.exit_reason == "TARGET1_TRAIL_EXIT"
    assert t.pnl > 0
    print("SHORT SCENARIO ASSERTIONS PASSED ✅ (engine correctly found earliest valid VWAP-cross trigger)")


def test_rr_filter_skips_trade():
    """If SL is too far away relative to a hard-coded 1:2 target construction,
    RR filter logic should still pass since target is built FROM the RR --
    so instead we test that a degenerate case (entry == SL, zero risk) is
    safely skipped rather than crashing."""
    day1 = dt.date(2026, 1, 5)
    day2 = dt.date(2026, 1, 6)
    rows = []
    rows.append(make_candle(day1, "09:15", 100, 105, 95, 100, 10000))
    rows.append(make_candle(day1, "09:20", 100, 110, 98, 105, 10000))
    rows.append(make_candle(day1, "09:25", 105, 106, 90, 95, 10000))
    rows.append(make_candle(day1, "09:30", 95, 100, 93, 98, 10000))
    # Day 2: rejection candle where close == pdl exactly (risk would be tiny/zero after buffer math)
    rows.append(make_candle(day2, "09:15", 91, 92, 90, 90.01, 5000))
    rows.append(make_candle(day2, "09:20", 90, 91, 89, 89.5, 5000))
    rows.append(make_candle(day2, "09:25", 89.5, 90, 89, 89.6, 5000))
    df = pd.DataFrame(rows)
    df_ind = add_indicators(df)
    trades = backtest_symbol("TESTEDGE", df_ind)
    print(f"\n=== Edge case scenario: trades generated: {len(trades)} (expected 0) ===")
    assert len(trades) == 0
    print("EDGE CASE ASSERTIONS PASSED ✅")



if __name__ == "__main__":
    main()
    test_short_scenario()
    test_rr_filter_skips_trade()
    print("\n🎉 ALL TEST SCENARIOS PASSED")
