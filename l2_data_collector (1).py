# l2_data_collector.py
# Description: Collects L2 order book data from Bybit via WebSocket and stores it in compressed files.

import ccxt
import pandas as pd
import time
import os
import json
from datetime import datetime, timezone
import websocket  # Make sure this is installed: pip install websocket-client
import threading
import gzip  # For compressing L2 data
import traceback  # For more detailed error logging
from typing import Any, Dict, Optional


class L2DataCollector:
    """
    Collects L2 order book data from Bybit via WebSocket and stores it in compressed files.
    """

    def __init__(self, config: Dict[str, Any], bot_base_dir: str) -> None:
        """
        Initialize the L2DataCollector.

        Args:
            config: Configuration dictionary.
            bot_base_dir: Base directory for bot data.
        """
        self.config = config  # Store the passed config
        self.symbol_config = config.get(
            'symbol', 'BTCUSDT'
        )  # Symbol from config for L2DataCollector specific usage
        self.exchange_name = config.get('exchange_name', 'bybit')
        # self.l2_limit = config.get('l2_fetch_limit', 1)  # This was more for REST, not primary for WS collector
        self.data_dir = os.path.join(
            bot_base_dir,
            config.get('l2_data_folder', 'l2_data')
        )
        self.log_file_path = os.path.join(
            bot_base_dir,
            config.get('l2_log_file', 'l2_data_collector.log')
        )
        self.max_file_size_mb = config.get(
            'l2_max_file_size_mb', 20
        )  # Using the more specific key
        self.collection_duration_seconds = config.get(
            'l2_collection_duration_seconds', 300
        )
        self.l2_websocket_depth = config.get(
            'l2_websocket_depth', 50
        )  # Specific key for WS depth

        self.terminate_event = threading.Event()
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.current_file_handler: Optional[gzip.GzipFile] = None
        self.current_file_path: Optional[str] = None
        self.file_start_time: Optional[float] = None

        os.makedirs(self.data_dir, exist_ok=True)

        try:
            # Initialize CCXT exchange object if needed for market info, but not strictly for WS connection itself
            # self.exchange = getattr(ccxt, self.exchange_name)()
            pass  # For now, L2 collector might not need a full ccxt exchange instance if WSS URL is directly constructed
        except AttributeError:
            self._log(
                f"Exchange {self.exchange_name} not found in CCXT (if it were used)."
            )
            # raise  # Not critical if we construct WSS URL manually
        except Exception as e:
            self._log(
                f"Error initializing exchange {self.exchange_name} (if it were used): {e}"
            )
            # raise

        self._log(
            f"L2DataCollector initialized for symbol: {self.symbol_config}, "
            f"exchange: {self.exchange_name}"
        )

    def _log(self, message: str) -> None:
        """
        Log a message to both stdout and the log file.

        Args:
            message: The message to log.
        """
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        log_message = f"[{timestamp}] (L2Collector) {message}"
        print(log_message)  # Print to notebook output as well
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(self.log_file_path)
            if log_dir and not os.path.exists(log_dir):  # Check if log_dir is not empty string
                os.makedirs(log_dir, exist_ok=True)

            with open(self.log_file_path, 'a') as f:
                f.write(log_message + '\n')
        except Exception as e:
            print(
                f"CRITICAL Error writing to L2 log file {self.log_file_path}: {e}"
            )

    def _get_new_file_path(self) -> str:
        """
        Generate a new file path for storing L2 data.

        Returns:
            The new file path as a string.
        """
        timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        # Use the symbol configured for the collector
        safe_collector_symbol = self.symbol_config.replace('/', '_').replace(':', '_')
        return os.path.join(
            self.data_dir,
            f'l2_data_{safe_collector_symbol}_{timestamp_str}.jsonl.gz'
        )

    def _open_new_file(self) -> None:
        """
        Open a new gzip file for writing L2 data.
        """
        if self.current_file_handler:
            try:
                self.current_file_handler.close()
                self._log(f"Closed file: {self.current_file_path}")
            except Exception as e:
                self._log(f"Error closing file {self.current_file_path}: {e}")

        self.current_file_path = self._get_new_file_path()
        try:
            self.current_file_handler = gzip.open(
                self.current_file_path, 'at', encoding='utf-8'
            )
            self.file_start_time = time.time()
            self._log(f"Opened new data file: {self.current_file_path}")
        except Exception as e:
            self._log(f"Error opening new file {self.current_file_path}: {e}")
            self.current_file_handler = None

    def _check_file_rotation(self) -> None:
        """
        Check if the current file needs to be rotated due to size.
        """
        if (
            self.current_file_handler and self.current_file_path and
            not self.current_file_handler.closed
        ):
            try:
                self.current_file_handler.flush()
                # os.fsync(self.current_file_handler.fileno())
                # fsync might error on gzip stream before close

                current_size_bytes = os.path.getsize(self.current_file_path)
                if current_size_bytes >= self.max_file_size_mb * 1024 * 1024:
                    self._log(
                        f"File {self.current_file_path} reached max size "
                        f"({current_size_bytes / (1024*1024):.2f} MB). Rotating."
                    )
                    self._open_new_file()
            except FileNotFoundError:
                # If file was just closed and path is being checked
                self._log(
                    f"File {self.current_file_path} not found during size check "
                    f"(likely just rotated)."
                )
            except Exception as e:
                self._log(
                    f"Error checking file size for {self.current_file_path}: {e}"
                )
                if not self.current_file_handler or self.current_file_handler.closed:
                    self._log(
                        f"File handler for {self.current_file_path} seems closed or "
                        f"invalid. Re-opening."
                    )
                    self._open_new_file()

    def _on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        """
        Handle incoming WebSocket messages.

        Args:
            ws: The WebSocketApp instance.
            message: The received message as a string.
        """
        try:
            data = json.loads(message)
            processed_data = None
            received_ts_ms = int(time.time() * 1000)

            if self.exchange_name == 'bybit':
                if (
                    'topic' in data and 'orderbook' in data['topic'] and
                    'data' in data
                ):
                    book_data = data['data']
                    exchange_ts_ms = data.get('ts', received_ts_ms)  # Prefer exchange timestamp
                    processed_data = {
                        'exchange': self.exchange_name,
                        'symbol': book_data.get('s', self.symbol_config),
                        'timestamp_ms': exchange_ts_ms,
                        'received_timestamp_ms': received_ts_ms,
                        'type': data.get('type', 'unknown'),
                        'bids': [
                            [float(p), float(q)]
                            for p, q in book_data.get('b', []) if p and q
                        ],  # Ensure p,q not None
                        'asks': [
                            [float(p), float(q)]
                            for p, q in book_data.get('a', []) if p and q
                        ],  # Ensure p,q not None
                        'update_id': book_data.get('u'),
                        'sequence_id': book_data.get('seq')
                    }
            else:
                self._log(
                    f"Received L2 data from unhandled exchange {self.exchange_name}, "
                    f"needs specific parsing logic."
                )
                processed_data = {
                    'exchange': self.exchange_name,
                    'symbol': self.symbol_config,
                    'timestamp_ms': received_ts_ms,
                    'raw_data': data
                }

            if (
                processed_data and self.current_file_handler and
                not self.current_file_handler.closed
            ):
                json_line = json.dumps(processed_data)
                self.current_file_handler.write(json_line + '\n')
                # self._check_file_rotation()  # Check rotation less frequently
            elif (
                not self.current_file_handler or
                (self.current_file_handler and self.current_file_handler.closed)
            ):
                self._log(
                    "File handler not available or closed. Message not saved. "
                    "Attempting to reopen."
                )
                self._open_new_file()
                if (
                    self.current_file_handler and
                    not self.current_file_handler.closed and processed_data
                ):
                    json_line = json.dumps(processed_data)
                    self.current_file_handler.write(json_line + '\n')

        except json.JSONDecodeError:
            self._log(
                f"Failed to decode JSON: {message[:200]}..."
            )  # Log only part of message
        except Exception as e:
            self._log(
                f"Error processing message: {e}, Message snippet: {message[:200]}..."
            )
            # traceback.print_exc()

    def _on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        """
        Handle WebSocket errors.

        Args:
            ws: The WebSocketApp instance.
            error: The error encountered.
        """
        self._log(f"WebSocket Error: {error}")
        if isinstance(error, ConnectionRefusedError):
            self._log("Connection was refused. Check network or WSS URL.")
            self.terminate_event.set()  # Stop trying if connection refused

    def _on_close(
        self, ws: websocket.WebSocketApp, close_status_code: int, close_msg: str
    ) -> None:
        """
        Handle WebSocket closure.

        Args:
            ws: The WebSocketApp instance.
            close_status_code: Status code for closure.
            close_msg: Closure message.
        """
        self._log(
            f"WebSocket Closed: Status={close_status_code}, Msg='{close_msg}'"
        )
        if not self.terminate_event.is_set():
            self._log(
                "WebSocket closed unexpectedly. Further action might be needed "
                "(e.g. reconnect)."
            )
            # For now, we let the collection end if the socket closes prematurely.

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        """
        Handle WebSocket opening and send subscription message.

        Args:
            ws: The WebSocketApp instance.
        """
        self._log(f"WebSocket Connection Opened for {self.symbol_config}")
        # Using self.l2_websocket_depth from config
        depth = self.l2_websocket_depth

        # Bybit uses the linear perpetual symbol for USDT margined contracts, e.g. BTCUSDT
        # symbol_config is expected to be in this format (e.g., 'BTCUSDT' from collector_symbol)
        ws_symbol = self.symbol_config.replace('/', '').split(':')[0]

        if self.exchange_name == 'bybit':
            subscription_msg = {
                "op": "subscribe",
                "args": [f"orderbook.{depth}.{ws_symbol}"]
            }
            try:
                ws.send(json.dumps(subscription_msg))
                self._log(f"Subscribed to orderbook.{depth}.{ws_symbol}")
            except Exception as e:
                self._log(f"Error sending subscription message: {e}")
                ws.close()  # Close if subscribe fails
        else:
            self._log(
                f"WebSocket subscription logic not implemented for {self.exchange_name}"
            )
            ws.close()

    def start_collection_websocket(self) -> None:
        """
        Start collecting L2 data via WebSocket.
        """
        market_type = self.config.get(
            'market_type', 'linear'
        )  # Get from L2 collector's config

        if self.exchange_name == 'bybit':
            if market_type not in ['linear', 'spot', 'inverse', 'option']:
                self._log(
                    f"Unsupported market_type '{market_type}' for Bybit WebSocket. "
                    f"Defaulting to 'linear'."
                )
                market_type = 'linear'
            wss_url = f"wss://stream.bybit.com/v5/public/{market_type}"
            self._log(f"Using WSS URL for Bybit ({market_type}): {wss_url}")
        else:
            self.terminate_event.set()  # Stop if not Bybit, as URL construction is specific
            self._log(
                f"Cannot determine WebSocket URL for {self.exchange_name}. "
                f"Stopping L2 collection."
            )
            return

        self._log("Starting WebSocket L2 data collection...")
        self._open_new_file()

        # websocket.enableTrace(True)  # For verbose WebSocket debugging
        self.ws = websocket.WebSocketApp(
            wss_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )

        self.ws_thread = threading.Thread(
            target=lambda: self.ws.run_forever(
                ping_interval=20, ping_timeout=10,
                sslopt={"check_hostname": False}
            )
        )  # Added sslopt for potential SSL issues
        self.ws_thread.daemon = True
        self.ws_thread.start()

        self._log(
            f"WebSocket collection thread started. Target duration: "
            f"{self.collection_duration_seconds} seconds."
        )

        start_time = time.time()
        last_rotation_check_time = start_time

        try:
            while (
                not self.terminate_event.is_set() and
                (time.time() - start_time) < self.collection_duration_seconds
            ):
                if not self.ws_thread.is_alive():
                    self._log(
                        "WebSocket thread died unexpectedly. Stopping collection."
                    )
                    break

                # Check for file rotation periodically (e.g., every 60 seconds)
                if time.time() - last_rotation_check_time > 60:
                    self._check_file_rotation()
                    last_rotation_check_time = time.time()

                time.sleep(0.5)  # Main loop check interval
            self._log("Collection duration reached or termination signaled.")
        except KeyboardInterrupt:
            self._log(
                "Keyboard interrupt received by L2 collector. Stopping collection."
            )
        finally:
            self.stop_collection_websocket()

    def stop_collection_websocket(self) -> None:
        """
        Stop the WebSocket L2 data collection and clean up resources.
        """
        self._log("Attempting to stop WebSocket L2 data collection...")
        self.terminate_event.set()
        if self.ws:
            try:
                self.ws.close()  # This should trigger _on_close
                self._log("WebSocket close request sent.")
            except Exception as e:
                self._log(f"Error during WebSocket explicit close: {e}")

        if self.ws_thread and self.ws_thread.is_alive():
            self._log("Waiting for WebSocket thread to join...")
            self.ws_thread.join(timeout=10.0)
            if self.ws_thread.is_alive():
                self._log("Warning: WebSocket thread did not join in time.")
            else:
                self._log("WebSocket thread joined.")

        if self.current_file_handler and not self.current_file_handler.closed:
            try:
                self.current_file_handler.close()
                self._log(f"Closed final data file: {self.current_file_path}")
            except Exception as e:
                self._log(
                    f"Error closing final data file {self.current_file_path}: {e}"
                )
        self._log("L2 data collection fully stopped.")

