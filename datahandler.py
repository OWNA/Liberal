# data_handler.py
# Reformatted from notebook export to standard Python file

import os
import time
import json
import gzip
import traceback
import pandas as pd
import ccxt  # For specific ccxt exceptions

class DataHandler:
    """
    Handles fetching, cleaning, loading, and aligning OHLCV and L2 order book data.
    """

    def __init__(self, config, exchange_api):
        """
        Initializes the DataHandler.

        Args:
            config (dict): Configuration dictionary.
            exchange_api: Initialized CCXT exchange object.
        """
        self.config = config
        self.exchange = exchange_api
        self.symbol = config.get('symbol', 'BTC/USDT')
        self.timeframe = config.get('timeframe', '1h')

        self.feature_window = config.get('feature_window', 50)
        self.l2_depth_live_snapshot = config.get('l2_depth', 25)

        self.base_dir = config.get('base_dir', './trading_bot_data')
        safe_symbol = self.symbol.replace('/', '_').replace(':', '')
        self.ohlcv_data_path = os.path.join(
            self.base_dir,
            f"ohlcv_data_{safe_symbol}_{self.timeframe}.csv"
        )

        collector_symbol_safe = config.get(
            'collector_symbol',
            self.symbol.replace('/', '').split(':')[0]
        ).lower()
        collector_duration = config.get('collector_duration', 5)
        collector_unit_char = config.get('collector_unit', 'minutes')[0]
        l2_collector_filename = (
            f"{collector_symbol_safe}_l2_data_{collector_duration}{collector_unit_char}.jsonl"
        )
        self.l2_raw_data_path = os.path.join(
            self.base_dir,
            config.get('l2_data_folder', 'l2_data'),
            l2_collector_filename
        )

        print(
            "DataHandler initialized. OHLCV path: "
            f"{self.ohlcv_data_path}, "
            "L2 Raw path: "
            f"{self.l2_raw_data_path}"
        )

    def fetch_ohlcv(self, limit=None, since=None, max_retries=3, delay_seconds=5):
        """
        Fetches historical OHLCV data from the exchange.
        """
        if not self.exchange:
            print("Error (DataHandler): Exchange API not available for fetching OHLCV.")
            return pd.DataFrame()

        limit = limit or self.config.get('fetch_ohlcv_limit', 1000)

        for attempt in range(max_retries):
            try:
                since_timestamp = since
                if since is None:
                    if hasattr(self.exchange, 'parse_timeframe') and hasattr(self.exchange, 'milliseconds'):
                        timeframe_duration_in_ms = self.exchange.parse_timeframe(self.timeframe) * 1000
                        buffer_candles = self.feature_window + 50
                        since_timestamp = self.exchange.milliseconds() - (limit + buffer_candles) * timeframe_duration_in_ms
                    else:
                        print("Warning (DataHandler): Exchange object missing parse_timeframe or milliseconds method. Cannot calculate 'since' timestamp accurately.")
                        # Fallback: fetch without 'since', relying on exchange default or 'limit' only
                        pass  # since_timestamp remains None

                print(f"Fetching OHLCV for {self.symbol}, timeframe {self.timeframe}, limit {limit}, since {pd.to_datetime(since_timestamp, unit='ms', utc=True) if since_timestamp else 'N/A'}")

                ohlcv_data = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, since=since_timestamp, limit=limit)

                if not ohlcv_data:
                    print(f"Warning (DataHandler): No OHLCV data returned (Attempt {attempt + 1}/{max_retries}).")
                    if attempt < max_retries - 1:
                        time.sleep(delay_seconds * (2**attempt))
                        continue
                    return pd.DataFrame()

                df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                print(f"Fetched {len(df)} OHLCV records.")
                return df.dropna()

            except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout, ccxt.RateLimitExceeded) as e:
                print(f"Warning (DataHandler): CCXT error fetching OHLCV (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = delay_seconds * (2**attempt)
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Error (DataHandler): Max retries reached for fetching OHLCV.")
                    return pd.DataFrame()
            except Exception as e:
                print(f"Error (DataHandler): Unexpected error fetching OHLCV: {e}")
                traceback.print_exc()
                return pd.DataFrame()
        return pd.DataFrame()

    def fetch_l2_order_book_snapshot(self, limit=None, max_retries=3, delay_seconds=2):
        """
        Fetches a *live snapshot* of the current Level 2 order book via REST API.
        """
        if not self.exchange or not self.exchange.has.get('fetchL2OrderBook'):
            print("Warning (DataHandler): Exchange does not support fetchL2OrderBook or API not available.")
            return None

        fetch_limit = limit if limit is not None else self.l2_depth_live_snapshot

        for attempt in range(max_retries):
            try:
                order_book = self.exchange.fetch_l2_order_book(self.symbol, limit=fetch_limit)
                if (order_book and isinstance(order_book, dict) and
                    'bids' in order_book and isinstance(order_book['bids'], list) and
                    'asks' in order_book and isinstance(order_book['asks'], list)):

                    if order_book['bids'] and (not isinstance(order_book['bids'][0], list) or len(order_book['bids'][0]) != 2):
                        print("Warning (DataHandler): Invalid L2 bids structure.")
                        return None
                    if order_book['asks'] and (not isinstance(order_book['asks'][0], list) or len(order_book['asks'][0]) != 2):
                        print("Warning (DataHandler): Invalid L2 asks structure.")
                        return None

                    order_book['fetch_timestamp_ms'] = self.exchange.milliseconds() if hasattr(self.exchange, 'milliseconds') else int(time.time() * 1000)
                    return order_book

                print(f"Warning (DataHandler): Invalid L2 data structure received (Attempt {attempt + 1}).")
                if attempt < max_retries - 1:
                    time.sleep(delay_seconds * (2**attempt))
                continue
            except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout, ccxt.RateLimitExceeded) as e:
                print(f"Warning (DataHandler): CCXT error fetching L2 snapshot (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay_seconds * (2**attempt))
            except Exception as e:
                print(f"Error (DataHandler): Unexpected error fetching L2 snapshot: {e}")
                traceback.print_exc()
                return None
        print("Error (DataHandler): Max retries reached for fetching L2 snapshot.")
        return None

    def clean_ohlcv_data(self, df):
        """
        Cleans the raw OHLCV DataFrame.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        df_cleaned = df.copy()

        if not pd.api.types.is_datetime64_any_dtype(df_cleaned['timestamp']):
            df_cleaned["timestamp"] = pd.to_datetime(df_cleaned["timestamp"], unit="ms", utc=True, errors='coerce')
        elif df_cleaned['timestamp'].dt.tz is None:
            df_cleaned['timestamp'] = df_cleaned['timestamp'].dt.tz_localize('utc')
        else:
            df_cleaned['timestamp'] = df_cleaned['timestamp'].dt.tz_convert('utc')

        df_cleaned.dropna(subset=['timestamp'], inplace=True)
        df_cleaned.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
        df_cleaned.sort_values('timestamp', inplace=True)
        df_cleaned.reset_index(drop=True, inplace=True)

        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        for col in ohlcv_cols:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

        df_cleaned.dropna(subset=ohlcv_cols, inplace=True)

        df_cleaned = df_cleaned[(df_cleaned['volume'] >= 0) &
                                (df_cleaned['open'] > 0) & (df_cleaned['high'] > 0) &
                                (df_cleaned['low'] > 0) & (df_cleaned['close'] > 0)]
        df_cleaned = df_cleaned[(df_cleaned['high'] >= df_cleaned['low']) &
                                (df_cleaned['high'] >= df_cleaned['open']) &
                                (df_cleaned['high'] >= df_cleaned['close']) &
                                (df_cleaned['low'] <= df_cleaned['open']) &
                                (df_cleaned['low'] <= df_cleaned['close'])]
        print(f"Cleaned OHLCV data. Shape: {df_cleaned.shape}")
        return df_cleaned

    def _load_historical_l2_data(self):
        """
        Loads historical L2 data from the .jsonl file.
        """
        if not os.path.exists(self.l2_raw_data_path):
            print(f"Warning (DataHandler): L2 raw data file not found at {self.l2_raw_data_path}")
            return pd.DataFrame()

        l2_data_list = []
        try:
            # Check if file is gzipped
            is_gzipped = self.l2_raw_data_path.endswith('.gz')
            open_func = gzip.open if is_gzipped else open
            read_mode = 'rt' if is_gzipped else 'r'  # text mode for reading lines

            with open_func(self.l2_raw_data_path, read_mode, encoding='utf-8') as f:
                for line in f:
                    try:
                        l2_data_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning (DataHandler): Skipping invalid JSON line in {self.l2_raw_data_path}")

            if not l2_data_list:
                print(f"Warning (DataHandler): No data loaded from {self.l2_raw_data_path}")
                return pd.DataFrame()

            df_l2 = pd.DataFrame(l2_data_list)
            if 'timestamp_ms' not in df_l2.columns:
                print(f"Warning (DataHandler): 'timestamp_ms' column not found in L2 data.")
                # Attempt to use 'received_timestamp_ms' or other timestamp fields if available
                if 'received_timestamp_ms' in df_l2.columns:
                    df_l2.rename(columns={'received_timestamp_ms': 'timestamp_ms'}, inplace=True)
                else:  # Add more fallbacks if needed
                    return pd.DataFrame()

            df_l2['timestamp'] = pd.to_datetime(df_l2['timestamp_ms'], unit='ms', utc=True, errors='coerce')
            df_l2.dropna(subset=['timestamp'], inplace=True)
            df_l2.sort_values('timestamp', inplace=True)
            print(f"Loaded {len(df_l2)} L2 records from {self.l2_raw_data_path}")
            # Ensure 'b' and 'a' columns exist, even if they are sometimes missing from a record
            if 'b' not in df_l2.columns: df_l2['b'] = None
            if 'a' not in df_l2.columns: df_l2['a'] = None
            return df_l2[['timestamp', 'b', 'a']]

        except Exception as e:
            print(f"Error (DataHandler): Failed to load or process L2 data from {self.l2_raw_data_path}: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def _align_l2_with_ohlcv(self, df_ohlcv, df_l2):
        """
        Aligns L2 snapshots with OHLCV data.
        """
        if df_ohlcv.empty or df_l2.empty:
            print("Warning (DataHandler): OHLCV or L2 DataFrame is empty. Cannot align.")
            df_ohlcv_aligned = df_ohlcv.copy()
            df_ohlcv_aligned['l2_bids'] = None
            df_ohlcv_aligned['l2_asks'] = None
            return df_ohlcv_aligned

        df_ohlcv = df_ohlcv.sort_values('timestamp').reset_index(drop=True)
        df_l2 = df_l2.sort_values('timestamp').reset_index(drop=True)

        df_ohlcv_aligned = pd.merge_asof(
            left=df_ohlcv,
            right=df_l2.rename(columns={'b': 'l2_bids', 'a': 'l2_asks'}),
            on='timestamp',
            direction='backward'
        )

        num_na_l2 = df_ohlcv_aligned['l2_bids'].isnull().sum()
        if num_na_l2 > 0:
            print(f"Warning (DataHandler): {num_na_l2} OHLCV candles could not be aligned with preceding L2 data.")

        print(f"Aligned L2 data with OHLCV. Resulting shape: {df_ohlcv_aligned.shape}")
        return df_ohlcv_aligned

    def load_and_prepare_historical_data(self, fetch_ohlcv_limit=None, use_historical_l2=False, save_ohlcv=True):
        """
        Orchestrates fetching/loading, cleaning, and aligning historical data.
        """
        df_ohlcv = None
        if os.path.exists(self.ohlcv_data_path) and self.config.get('load_existing_ohlcv', True):
            try:
                print(f"Loading existing OHLCV data from {self.ohlcv_data_path}")
                df_ohlcv = pd.read_csv(self.ohlcv_data_path, parse_dates=['timestamp'])
                if df_ohlcv['timestamp'].dt.tz is None:
                    df_ohlcv['timestamp'] = df_ohlcv['timestamp'].dt.tz_localize('utc')
                else:
                    df_ohlcv['timestamp'] = df_ohlcv['timestamp'].dt.tz_convert('utc')
                print(f"Loaded {len(df_ohlcv)} records from CSV.")
            except Exception as e:
                print(f"Warning (DataHandler): Could not load {self.ohlcv_data_path}, fetching fresh data. Error: {e}")
                df_ohlcv = None

        if df_ohlcv is None or df_ohlcv.empty:
            print("Fetching fresh OHLCV data...")
            df_ohlcv = self.fetch_ohlcv(limit=fetch_ohlcv_limit)
            if save_ohlcv and not df_ohlcv.empty:
                try:
                    os.makedirs(self.base_dir, exist_ok=True)
                    df_ohlcv.to_csv(self.ohlcv_data_path, index=False)
                    print(f"Saved fetched OHLCV data to {self.ohlcv_data_path}")
                except Exception as e:
                    print(f"Warning (DataHandler): Could not save OHLCV data: {e}")

        if df_ohlcv is None or df_ohlcv.empty:
            print("Error (DataHandler): Failed to obtain OHLCV data.")
            return pd.DataFrame()

        df_ohlcv_clean = self.clean_ohlcv_data(df_ohlcv)
        if df_ohlcv_clean.empty:
            print("Error (DataHandler): OHLCV data is empty after cleaning.")
            return pd.DataFrame()

        df_processed = df_ohlcv_clean
        if use_historical_l2:
            print("Loading and aligning historical L2 data...")
            df_l2_historical = self._load_historical_l2_data()
            if not df_l2_historical.empty:
                df_processed = self._align_l2_with_ohlcv(df_ohlcv_clean, df_l2_historical)
            else:
                print("Warning (DataHandler): Historical L2 data is empty. Proceeding without L2 alignment.")
                # Ensure columns exist if L2 was expected but not found
                if 'l2_bids' not in df_processed.columns: df_processed['l2_bids'] = None
                if 'l2_asks' not in df_processed.columns: df_processed['l2_asks'] = None
        else: # If not using historical L2, ensure columns are present as None for consistency
            if 'l2_bids' not in df_processed.columns: df_processed['l2_bids'] = None
            if 'l2_asks' not in df_processed.columns: df_processed['l2_asks'] = None

        print(f"Data loading and preparation complete. Final DataFrame shape: {df_processed.shape}")
        return df_processed