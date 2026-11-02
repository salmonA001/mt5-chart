"""Serve live MT5 bars and a forecast scanner to the chart page.

Run on the Windows machine where MetaTrader 5 is open and logged in:
    pip install MetaTrader5
    python server.py
Then open http://127.0.0.1:8765
"""
import json
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import MetaTrader5 as mt5
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

PORT = 8765
BARS = 700  # same history length the chart page requests, so scan and chart backtests match
PAGE = Path(__file__).with_name("index.html").read_bytes()
TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


def get_bars(symbol: str, tf: str, count: int) -> dict:
    if tf not in TIMEFRAMES:
        raise ValueError(f"Unknown timeframe {tf}. Use one of {', '.join(TIMEFRAMES)}")
    if not mt5.symbol_select(symbol, True):
        raise ValueError(f"Symbol {symbol} not found at this broker (check Market Watch for the exact name, e.g. XAUUSDm)")
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAMES[tf], 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"MT5 returned no bars for {symbol} {tf}: {mt5.last_error()}")
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"MT5 returned no tick for {symbol}: {mt5.last_error()}")
    bars = [
        {"time": int(r["time"]), "open": float(r["open"]), "high": float(r["high"]),
         "low": float(r["low"]), "close": float(r["close"])}
        for r in rates
    ]
    return {"bars": bars, "bid": tick.bid, "ask": tick.ask}


def all_symbols() -> tuple:
    symbols = mt5.symbols_get()
    if symbols is None:
        raise RuntimeError(f"MT5 returned no symbol list: {mt5.last_error()}")
    return symbols


def group_of(path: str) -> str:
    return path.rsplit("\\", 1)[0]


def get_groups() -> list:
    counts = Counter(group_of(s.path) for s in all_symbols())
    return [{"group": g, "count": n} for g, n in sorted(counts.items())]


def forecast(closes: np.ndarray, n: int, h: int) -> dict:
    """Linear-regression forecast h bars ahead from the last n closes, plus a walk-forward backtest."""
    x = np.arange(n) - (n - 1) / 2
    sxx = float(x @ x)
    win = sliding_window_view(closes, n)
    slope = win @ x / sxx
    mean = win.mean(axis=1)
    sd = np.sqrt(((win - mean[:, None] - slope[:, None] * x) ** 2).sum(axis=1) / (n - 2))
    xf = (n - 1) / 2 + h
    pred = mean + slope * xf
    half = 1.96 * sd * np.sqrt(1 + 1 / n + xf ** 2 / sxx)
    k = len(closes) - n - h + 1  # windows whose outcome is already known
    actual = closes[n - 1 + h:]
    hit = np.sign(slope[:k]) == np.sign(actual - closes[n - 1:n - 1 + k])
    inside = np.abs(actual - pred[:k]) <= half[:k]
    last = float(closes[-1])
    return {"last": last, "forecast": float(pred[-1]), "change_pct": float((pred[-1] - last) / last * 100),
            "hit": float(hit.mean()), "band": float(inside.mean()), "tests": int(k)}


def scan(group: str, tf: str, n: int, h: int) -> dict:
    if tf not in TIMEFRAMES:
        raise ValueError(f"Unknown timeframe {tf}. Use one of {', '.join(TIMEFRAMES)}")
    symbols = [s.name for s in all_symbols() if group_of(s.path) == group]
    if not symbols:
        raise ValueError(f"No symbols in group {group}")
    rows, skipped = [], []
    for name in symbols:
        mt5.symbol_select(name, True)
        rates = mt5.copy_rates_from_pos(name, TIMEFRAMES[tf], 0, BARS)
        if rates is None or len(rates) < n + h + 30:
            skipped.append(name)
            continue
        rows.append({"symbol": name, **forecast(rates["close"].astype(float), n, h)})
    return {"rows": rows, "skipped": skipped}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self.send(200, PAGE, "text/html; charset=utf-8")
        elif url.path.startswith("/api/"):
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/api/symbols":
                    data = {"symbols": sorted(s.name for s in all_symbols())}
                elif url.path == "/api/groups":
                    data = {"groups": get_groups()}
                elif url.path == "/api/bars":
                    data = get_bars(q["symbol"], q["tf"], int(q["count"]))
                elif url.path == "/api/scan":
                    data = scan(q["group"], q["tf"], int(q["lookback"]), int(q["horizon"]))
                else:
                    raise ValueError(f"Unknown endpoint {url.path}")
                self.send_json(200, data)
            except (ValueError, RuntimeError) as e:
                self.send_json(400, {"error": str(e)})
        else:
            self.send(404, b"not found", "text/plain")

    def do_OPTIONS(self):
        # CORS/private-network preflight so a hosted copy of the page can read this local server
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()

    def send_json(self, code, obj):
        self.send(code, json.dumps(obj).encode(), "application/json")

    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    if not mt5.initialize():
        raise SystemExit(f"Cannot connect to MetaTrader 5 (is it open and logged in?): {mt5.last_error()}")
    info = mt5.account_info()
    if info is None:
        raise SystemExit(f"MetaTrader 5 is open but not logged in to an account: {mt5.last_error()}")
    print(f"Connected to MT5 account {info.login} on {info.server}")
    print(f"Open http://127.0.0.1:{PORT}")
    # Single-threaded on purpose: the MetaTrader5 package is not thread-safe.
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
