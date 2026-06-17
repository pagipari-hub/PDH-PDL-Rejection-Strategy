"""
Test for the incremental CSV candle cache (load_or_fetch_all_symbols) and
the rate-limit detection helper, using a mocked Angel One API so no real
network/API access is needed.
"""
import os
import shutil
import datetime as dt
import pandas as pd

import pdh_pdl_backtest as mod
from pdh_pdl_backtest import (
    Config, load_or_fetch_all_symbols, _is_rate_limit_response,
    fetch_historical_candles,
)


class FakeSmartConnectFullHistory:
    """Simulates Angel One returning candles for whatever date range it's
    asked for, with a tiny fixed call counter so we can assert on request
    counts (the whole point of this refactor)."""
    def __init__(self):
        self.call_count = 0

    def getCandleData(self, params):
        self.call_count += 1
        from_dt = dt.datetime.strptime(params["fromdate"], "%Y-%m-%d %H:%M")
        to_dt = dt.datetime.strptime(params["todate"], "%Y-%m-%d %H:%M")

        rows = []
        cur = from_dt
        while cur < to_dt:
            if cur.weekday() < 5 and dt.time(9, 15) <= cur.time() <= dt.time(15, 25):
                rows.append([
                    cur.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
                    100.0, 101.0, 99.0, 100.5, 1000,
                ])
            cur += dt.timedelta(minutes=5)

        return {"status": True, "data": rows}


class FakeSmartConnectRateLimited:
    """Fails the first N calls with a rate-limit-style response, then
    succeeds, to test the retry/backoff path without a real 45s sleep."""
    def __init__(self, fail_times=2):
        self.call_count = 0
        self.fail_times = fail_times

    def getCandleData(self, params):
        self.call_count += 1
        if self.call_count <= self.fail_times:
            return {"status": False, "message": "Access denied because of exceeding access rate"}
        return {"status": True, "data": [
            [params["fromdate"].replace(" ", "T") + ":00+05:30", 100, 101, 99, 100.5, 1000]
        ]}


def test_rate_limit_detection():
    assert _is_rate_limit_response({"status": False, "message": "Access denied because of exceeding access rate"})
    assert _is_rate_limit_response("Too many requests")
    assert not _is_rate_limit_response({"status": True, "data": []})
    print("RATE LIMIT DETECTION TEST PASSED ✅")


def test_rate_limit_retry_path(monkeypatch):
    fake = FakeSmartConnectRateLimited(fail_times=2)
    # speed up the test: patch sleep so we don't actually wait 45s x 2
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    from_dt = dt.datetime(2026, 1, 5, 9, 15)
    to_dt = dt.datetime(2026, 1, 5, 9, 20)
    df = fetch_historical_candles(fake, "TESTTOKEN", from_dt, to_dt)

    assert fake.call_count == 3, f"Expected 3 calls (2 rate-limited + 1 success), got {fake.call_count}"
    assert not df.empty
    print("RATE LIMIT RETRY PATH TEST PASSED ✅ (call_count=%d)" % fake.call_count)


def test_incremental_cache_first_run_then_gap_only():
    test_cache_dir = "test_candle_cache_tmp"
    Config.CANDLE_CACHE_DIR = test_cache_dir
    if os.path.exists(test_cache_dir):
        shutil.rmtree(test_cache_dir)

    try:
        fake = FakeSmartConnectFullHistory()
        token_map = {"TESTSYM": "12345"}

        # --- First run: no cache exists, should fetch full lookback window ---
        data1 = load_or_fetch_all_symbols(fake, token_map, lookback_days=10)
        first_run_calls = fake.call_count
        assert "TESTSYM" in data1
        assert first_run_calls >= 1, "Expected at least one API call on first run"
        assert os.path.exists(os.path.join(test_cache_dir, "TESTSYM.csv"))
        first_run_rows = len(data1["TESTSYM"])
        print(f"First run: {first_run_calls} API call(s), {first_run_rows} candle rows cached")

        # --- Second run (immediately after): cache should already be up to date,
        #     since FakeSmartConnect generates candles right up to 'to_dt' which
        #     load_or_fetch computes as today at 15:30. No new API calls expected
        #     UNLESS 'now' has moved into a new 5-min bucket, so we just assert
        #     it's much cheaper than the first run (not equal-or-more requests). ---
        calls_before_second_run = fake.call_count
        data2 = load_or_fetch_all_symbols(fake, token_map, lookback_days=10)
        second_run_calls = fake.call_count - calls_before_second_run

        assert second_run_calls <= 1, (
            f"Expected at most 1 incremental API call on second run, got {second_run_calls}"
        )
        print(f"Second run: {second_run_calls} additional API call(s) (incremental gap-fetch)")

        assert len(data2["TESTSYM"]) >= first_run_rows, "Cache should not lose rows between runs"

        print("INCREMENTAL CACHE TEST PASSED ✅ "
              f"(first_run_calls={first_run_calls}, second_run_calls={second_run_calls})")
    finally:
        if os.path.exists(test_cache_dir):
            shutil.rmtree(test_cache_dir)


def test_instrument_master_csv_cache(monkeypatch, tmp_path=None):
    """Verifies the instrument master is read from CSV cache on a second
    call within the freshness window, without re-downloading."""
    cache_file = "test_instrument_master_tmp.csv"
    Config.INSTRUMENT_MASTER_CACHE_PATH = cache_file
    if os.path.exists(cache_file):
        os.remove(cache_file)

    download_calls = {"count": 0}

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload
        def read(self):
            import json
            return json.dumps(self._payload).encode("utf-8")
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def fake_urlopen(url, timeout=60):
        download_calls["count"] += 1
        return FakeResp([
            {"token": "1", "symbol": "RELIANCE-EQ", "exch_seg": "NSE", "name": "RELIANCE", "lotsize": "1"},
        ])

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    try:
        df1 = mod.load_instrument_master()
        assert download_calls["count"] == 1
        df2 = mod.load_instrument_master()
        assert download_calls["count"] == 1, "Second call within freshness window should use cache, not re-download"
        assert len(df1) == len(df2) == 1
        print("INSTRUMENT MASTER CACHE TEST PASSED ✅ (downloaded once, served from cache thereafter)")
    finally:
        if os.path.exists(cache_file):
            os.remove(cache_file)


if __name__ == "__main__":
    # lightweight monkeypatch shim since we're not using pytest here
    class _MonkeyPatch:
        def __init__(self):
            self._patches = []
        def setattr(self, obj, name, value):
            old = getattr(obj, name)
            self._patches.append((obj, name, old))
            setattr(obj, name, value)
        def undo(self):
            for obj, name, old in self._patches:
                setattr(obj, name, old)

    test_rate_limit_detection()

    mp = _MonkeyPatch()
    try:
        test_rate_limit_retry_path(mp)
    finally:
        mp.undo()

    test_incremental_cache_first_run_then_gap_only()

    mp2 = _MonkeyPatch()
    try:
        test_instrument_master_csv_cache(mp2)
    finally:
        mp2.undo()

    print("\n🎉 ALL DATA-LAYER TESTS PASSED")
