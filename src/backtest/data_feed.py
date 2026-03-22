"""Data acquisition and caching for backtesting.

Fetches historical price data via ``yfinance`` and converts it into
``backtrader``-compatible data feeds.  Responses are cached locally so
repeated runs for the same symbol / range avoid network calls.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import backtrader as bt
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Default cache location relative to project root.
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "backtest_cache"


class DataFeedProvider:
    """Download and cache OHLCV data, then convert to backtrader feeds.

    Parameters
    ----------
    cache_dir:
        Directory for cached CSV files.  Defaults to
        ``<project>/data/backtest_cache/``.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stock_data(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch stock OHLCV data via yfinance (cached).

        Parameters
        ----------
        symbol:
            Ticker symbol, e.g. ``"AAPL"`` or ``"MSFT"``.
        start:
            Start date in ``YYYY-MM-DD`` format.
        end:
            End date in ``YYYY-MM-DD`` format.
        interval:
            Bar interval — ``"1d"``, ``"1h"``, ``"5m"``, etc.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``Open, High, Low, Close, Volume``
            indexed by ``datetime``.
        """
        return self._fetch(symbol, start, end, interval)

    def get_crypto_data(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch cryptocurrency OHLCV data via yfinance (cached).

        Appends ``-USD`` to the symbol if not already present so that
        ``"BTC"`` becomes ``"BTC-USD"`` on Yahoo Finance.

        Parameters
        ----------
        symbol:
            Crypto symbol, e.g. ``"BTC"`` or ``"ETH-USD"``.
        start:
            Start date in ``YYYY-MM-DD`` format.
        end:
            End date in ``YYYY-MM-DD`` format.
        interval:
            Bar interval — ``"1d"``, ``"1h"``, ``"5m"``, etc.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``Open, High, Low, Close, Volume``
            indexed by ``datetime``.
        """
        if "-" not in symbol:
            symbol = f"{symbol}-USD"
        return self._fetch(symbol, start, end, interval)

    def to_backtrader_feed(self, df: pd.DataFrame) -> bt.feeds.PandasData:
        """Convert a pandas OHLCV DataFrame to a backtrader data feed.

        The DataFrame must have a ``DatetimeIndex`` and columns
        ``Open, High, Low, Close, Volume``.

        Parameters
        ----------
        df:
            Price data returned by :meth:`get_stock_data` or
            :meth:`get_crypto_data`.

        Returns
        -------
        bt.feeds.PandasData
            A backtrader-compatible data feed ready for ``cerebro.adddata``.
        """
        # Normalise column names to title case expected by backtrader
        col_map: dict[str, str] = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ("open", "high", "low", "close", "volume", "adj close"):
                col_map[col] = col.capitalize()
        if col_map:
            df = df.rename(columns=col_map)

        # Ensure a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df = df.set_index("Date")
                df.index = pd.to_datetime(df.index)
            elif "Datetime" in df.columns:
                df = df.set_index("Datetime")
                df.index = pd.to_datetime(df.index)
            else:
                df.index = pd.to_datetime(df.index)

        # Drop timezone info if present (backtrader prefers naive datetimes)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        feed = bt.feeds.PandasData(dataname=df)
        return feed

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, symbol: str, start: str, end: str, interval: str) -> Path:
        """Build a deterministic cache file path."""
        key = f"{symbol}_{start}_{end}_{interval}"
        hashed = hashlib.sha256(key.encode()).hexdigest()[:12]
        safe_symbol = symbol.replace("/", "_").replace("\\", "_")
        filename = f"{safe_symbol}_{start}_{end}_{interval}_{hashed}.csv"
        return self.cache_dir / filename

    def _load_cache(self, path: Path) -> pd.DataFrame | None:
        """Load cached CSV if it exists and is non-empty."""
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if df.empty:
                return None
            logger.debug("Loaded cached data from %s (%d rows)", path, len(df))
            return df
        except Exception:
            logger.warning("Failed to read cache file %s", path, exc_info=True)
            return None

    def _save_cache(self, df: pd.DataFrame, path: Path) -> None:
        """Write DataFrame to a CSV cache file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path)
            logger.debug("Saved %d rows to cache %s", len(df), path)
        except Exception:
            logger.warning("Failed to save cache to %s", path, exc_info=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch(self, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
        """Download data from yfinance with caching."""
        cache = self._cache_path(symbol, start, end, interval)
        cached_df = self._load_cache(cache)
        if cached_df is not None:
            return cached_df

        logger.info("Downloading %s data for %s (%s to %s)", interval, symbol, start, end)
        ticker = yf.Ticker(symbol)
        df: pd.DataFrame = ticker.history(start=start, end=end, interval=interval)

        if df.empty:
            raise ValueError(
                f"No data returned for {symbol} from {start} to {end} "
                f"(interval={interval}). Check the symbol and date range."
            )

        # Keep only standard OHLCV columns (drop Dividends, Stock Splits, etc.)
        keep_cols = [c for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")]
        df = df[keep_cols]

        # Strip timezone info to avoid mixed-timezone errors
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        self._save_cache(df, cache)
        logger.info("Downloaded %d bars for %s", len(df), symbol)
        return df
