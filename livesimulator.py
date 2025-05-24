# live_simulator.py
# Reformatted from notebook export to standard Python file

import os
import threading
import traceback
import pandas as pd
import ccxt

# Assumed to be imported in the main script and passed to the constructor
# from data_handler import DataHandler
# from feature_engineer import FeatureEngineer
# from model_predictor import ModelPredictor
# from advanced_risk_manager import AdvancedRiskManager
# from smart_order_executor import SmartOrderExecutor

class LiveSimulator:
    """
    Manages the live simulation (paper trading) loop.
    """

    def __init__(self, config, exchange_api, data_handler, feature_engineer,
                 model_predictor, risk_manager, order_executor):
        """
        Initializes the LiveSimulator.

        Args:
            config (dict): Configuration dictionary.
            exchange_api: Initialized CCXT exchange object.
            data_handler (DataHandler): Instance of DataHandler.
            feature_engineer (FeatureEngineer): Instance of FeatureEngineer.
            model_predictor (ModelPredictor): Instance of ModelPredictor.
            risk_manager (AdvancedRiskManager): Instance of AdvancedRiskManager.
            order_executor (SmartOrderExecutor): Instance of SmartOrderExecutor.
        """
        self.config = config
        self.exchange = exchange_api
        self.data_handler = data_handler
        self.feature_engineer = feature_engineer
        self.model_predictor = model_predictor
        self.risk_manager = risk_manager
        self.order_executor = order_executor

        self.symbol = config.get('symbol', 'BTC/USDT')
        self.timeframe = config.get('timeframe', '1h')
        self.base_dir = config.get('base_dir', './trading_bot_data')

        self.live_equity_history = []
        self.simulation_running = False
        self.simulation_stop_event = threading.Event()
        self.simulation_thread = None
        self.current_live_position = {
            "side": None, "entry_price": 0.0, "size": 0.0, "timestamp": None,
            "sl_price": None, "tp_price": None, "entry_commission": 0.0
        }

        safe_symbol = self.symbol.replace('/', '_').replace(':', '')
        self.simulation_log_path = os.path.join(self.base_dir, f"simulation_log_{safe_symbol}_{self.timeframe}.jsonl")

        self.use_l2_features_in_sim = config.get('use_l2_features', False) and \
                                      self.exchange and \
                                      self.exchange.has.get('fetchL2OrderBook')

        self.HAS_PANDAS_TA = getattr(feature_engineer, 'HAS_PANDAS_TA', False) if feature_engineer else False
        self.ta = getattr(feature_engineer, 'ta', None) if feature_engineer else None

        print("LiveSimulator initialized.")

    def _log_simulation_action(self, timestamp, action_type, details):
        """Logs simulation actions to a JSONL file."""
        try:
            ts_iso = pd.Timestamp(timestamp).tz_convert('UTC').isoformat() if pd.Timestamp(timestamp).tzinfo else \
                     pd.Timestamp(timestamp).tz_localize('UTC').isoformat()

            def convert_numpy_types(obj):
                if isinstance(obj, np.generic): return obj.item()
                if isinstance(obj, pd.Timestamp): return obj.isoformat()
                if isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
                return obj

            details_serializable = {k: convert_numpy_types(v) for k, v in details.items()}
            log_entry = {"timestamp": ts_iso, "action": action_type, **details_serializable}

            os.makedirs(os.path.dirname(self.simulation_log_path), exist_ok=True)
            with open(self.simulation_log_path, "a") as f:
                json.dump(log_entry, f)
                f.write("\n")
        except Exception as e:
            print(f"Error (LiveSimulator) logging sim action: {e}. Details: {details}")
            traceback.print_exc(limit=1)

    def _simulation_loop(self, initial_equity, threshold, fetch_limit,
                         loop_interval_seconds, commission_pct, leverage):
        """Core loop for the live simulation thread."""
        print(f"--- Live Simulation Thread Started ---")
        print(f"Params: InitEq={initial_equity}, Thresh={threshold}, FetchLimit={fetch_limit}, Interval={loop_interval_seconds}s, Comm={commission_pct*100:.4f}%, Lev={leverage}x")

        self.live_equity_history = [{'timestamp': datetime.now(timezone.utc), 'equity': initial_equity}]
        self.current_live_position = {
            "side": None, "entry_price": 0.0, "size": 0.0, "timestamp": None,
            "sl_price": None, "tp_price": None, "entry_commission": 0.0
        }
        balance = initial_equity

        use_ensemble_for_sim = self.config.get('use_ensemble_for_simulation', False)
        if not self.model_predictor.model_object:
            if not self.model_predictor.load_model_and_features(load_ensemble=use_ensemble_for_sim):
                print("FATAL (LiveSimulator): Failed to load model for simulation.")
                self._log_simulation_action(datetime.now(timezone.utc), "ERROR_SIM_SETUP", {"message": "Failed to load model"})
                self.simulation_running = False
                return

        # Ensure scaling info is available in ModelPredictor
        # This should have been set if model was trained in same session, or loaded with model.
        # _ensure_scaling_info will try to recalculate if missing.
        if not self.model_predictor._ensure_scaling_info():
             print("Warning (LiveSimulator): Scaling info for ModelPredictor could not be confirmed. Predictions might be unscaled or use defaults.")

        while self.simulation_running and not self.simulation_stop_event.is_set():
            loop_start_time = time.monotonic()
            now_utc = datetime.now(timezone.utc)
            try:
                required_candles = self.feature_engineer.feature_window + 50 if self.feature_engineer else 100
                df_raw_ohlcv = self.data_handler.fetch_ohlcv(limit=max(fetch_limit, required_candles))

                if df_raw_ohlcv is None or df_raw_ohlcv.empty or len(df_raw_ohlcv) < required_candles:
                    print(f"[{now_utc.strftime('%H:%M:%S')}] Insufficient OHLCV ({len(df_raw_ohlcv) if df_raw_ohlcv is not None else 0}). Waiting...")
                    self._log_simulation_action(now_utc, "WAIT_OHLCV", {"count": len(df_raw_ohlcv) if df_raw_ohlcv is not None else 0})
                    time.sleep(max(0.1, loop_interval_seconds - (time.monotonic() - loop_start_time))) # Ensure at least 0.1s sleep
                    continue

                df_cleaned_ohlcv = self.data_handler.clean_ohlcv_data(df_raw_ohlcv)
                if df_cleaned_ohlcv.empty:
                    time.sleep(max(0.1, loop_interval_seconds - (time.monotonic() - loop_start_time)))
                    continue

                df_for_features = df_cleaned_ohlcv.copy()
                l2_log_features = {}
                if self.use_l2_features_in_sim and self.feature_engineer:
                    l2_snapshot = self.data_handler.fetch_l2_order_book_snapshot()
                    if l2_snapshot and 'bids' in l2_snapshot and 'asks' in l2_snapshot:
                        l2_calculated_features = self.feature_engineer.calculate_l2_features_from_snapshot(
                            l2_snapshot['bids'], l2_snapshot['asks']
                        )
                        l2_log_features = {k: v for k, v in l2_calculated_features.items() if pd.notna(v)}
                        last_idx = df_for_features.index[-1]
                        for f_name, f_val in l2_calculated_features.items():
                            if f_name in self.feature_engineer.l2_features_list_config:
                                df_for_features.loc[last_idx, f_name] = f_val
                    else:
                        last_idx = df_for_features.index[-1]
                        for f_name in self.feature_engineer.l2_features_list_config:
                            df_for_features.loc[last_idx, f_name] = np.nan

                df_with_all_features = self.feature_engineer.generate_all_features(df_for_features, save=False) if self.feature_engineer else df_for_features

                if df_with_all_features.empty:
                    time.sleep(max(0.1, loop_interval_seconds - (time.monotonic() - loop_start_time)))
                    continue

                latest_features_df = df_with_all_features.iloc[-1:].copy()
                if latest_features_df.empty:
                    time.sleep(max(0.1, loop_interval_seconds - (time.monotonic() - loop_start_time)))
                    continue

                # Ensure ATR for risk management
                if 'atr' not in latest_features_df.columns or pd.isna(latest_features_df['atr'].iloc[0]):
                    if self.HAS_PANDAS_TA and self.ta and 'high' in df_with_all_features and \
                       'low' in df_with_all_features and 'close' in df_with_all_features:
                        atr_lookback = self.config.get('risk_management', {}).get('volatility_lookback', 14)
                        atr_series = self.ta.atr(df_with_all_features['high'], df_with_all_features['low'], df_with_all_features['close'], length=atr_lookback)
                        if atr_series is not None and not atr_series.empty:
                            latest_features_df['atr'] = atr_series.iloc[-1]
                        else: latest_features_df['atr'] = 0.01 * latest_features_df['close'].iloc[0]
                    else: latest_features_df['atr'] = 0.01 * latest_features_df['close'].iloc[0]
                if pd.isna(latest_features_df['atr'].iloc[0]):
                    latest_features_df['atr'] = 0.01 * latest_features_df['close'].iloc[0]

                predicted_df = self.model_predictor.predict_signals(
                    latest_features_df,
                    use_ensemble=use_ensemble_for_sim
                )
                if predicted_df is None or predicted_df.empty:
                    self._log_simulation_action(now_utc, "WAIT_PREDICTION", {})
                    time.sleep(max(0.1, loop_interval_seconds - (time.monotonic() - loop_start_time)))
                    continue

                current_signal = predicted_df["signal"].iloc[0]
                current_price = predicted_df["close"].iloc[0]
                current_timestamp = pd.Timestamp(predicted_df["timestamp"].iloc[0])
                current_atr = latest_features_df["atr"].iloc[0]
                pred_scaled = predicted_df["pred_scaled"].iloc[0] if "pred_scaled" in predicted_df.columns else np.nan
                pred_unscaled = predicted_df.get("pred_unscaled_target", pd.Series([np.nan])).iloc[0]

                pos = self.current_live_position
                unrealized_pnl = 0
                if pos["side"] == "long": unrealized_pnl = (current_price - pos["entry_price"]) * pos["size"]
                elif pos["side"] == "short": unrealized_pnl = (pos["entry_price"] - current_price) * pos["size"]

                current_equity = balance + unrealized_pnl
                self.live_equity_history.append({'timestamp': current_timestamp, 'equity': current_equity})

                exit_reason, exit_price = None, current_price
                last_high, last_low = predicted_df["high"].iloc[0], predicted_df["low"].iloc[0]

                if pos["side"] == "long" and pos["sl_price"] and last_low <= pos["sl_price"]:
                    exit_reason, exit_price, current_signal = "SL_HIT", pos["sl_price"], -1
                elif pos["side"] == "long" and pos["tp_price"] and last_high >= pos["tp_price"]:
                    exit_reason, exit_price, current_signal = "TP_HIT", pos["tp_price"], -1
                elif pos["side"] == "short" and pos["sl_price"] and last_high >= pos["sl_price"]:
                    exit_reason, exit_price, current_signal = "SL_HIT", pos["sl_price"], 1
                elif pos["side"] == "short" and pos["tp_price"] and last_low <= pos["tp_price"]:
                    exit_reason, exit_price, current_signal = "TP_HIT", pos["tp_price"], 1

                executed_action = "HOLD" if pos["side"] else "FLAT"
                log_details = {
                    "price": float(current_price), "signal": int(current_signal),
                    "pred_scaled": float(pred_scaled) if pd.notna(pred_scaled) else None,
                    "pred_unscaled_target": float(pred_unscaled) if pd.notna(pred_unscaled) else None,
                    "atr": float(current_atr) if pd.notna(current_atr) else None,
                    "equity_before": float(current_equity), **l2_log_features
                }

                pos_side_int = 1 if pos["side"] == "long" else -1 if pos["side"] == "short" else 0
                if pos["side"] and (current_signal == -pos_side_int or current_signal == 0 or exit_reason):
                    if self.order_executor:
                        self.order_executor.execute_order(
                            self.symbol, "sell" if pos["side"] == "long" else "buy",
                            pos["size"], exit_price, 'limit'
                        )

                    pnl_asset = (exit_price - pos["entry_price"]) if pos["side"] == "long" else (pos["entry_price"] - exit_price)
                    pnl_gross = pnl_asset * pos["size"]
                    commission_exit_cost = abs(exit_price * pos["size"] * commission_pct)
                    pnl_net = pnl_gross - pos.get("entry_commission", 0) - commission_exit_cost
                    balance += pnl_net

                    executed_action = f"EXIT_{pos['side'].upper()}"
                    log_details.update({
                        "exit_reason": exit_reason or "SignalFlipOrNeutral",
                        "entry_price": pos["entry_price"], "exit_price": exit_price,
                        "size": pos["size"], "pnl_net_trade": pnl_net, "final_balance": balance
                    })
                    self.current_live_position = {"side": None, "entry_price": 0.0, "size": 0.0, "timestamp": None, "sl_price": None, "tp_price": None, "entry_commission": 0.0}

                elif not pos["side"] and current_signal != 0 and not exit_reason:
                    entry_side = "long" if current_signal == 1 else "short"
                    volatility_pct = (current_atr / current_price) if current_price > 0 and pd.notna(current_atr) and current_atr > 1e-9 else \
                                     self.config.get('fallback_volatility_pct_for_sizing', 0.02)

                    target_position_size_usd = self.risk_manager.calculate_position_size(current_equity, volatility_pct) if self.risk_manager else current_equity * 0.01 # Fallback if no risk manager
                    size_asset = (target_position_size_usd * leverage) / current_price if current_price > 0 else 0

                    if size_asset > 1e-8:
                        entry_commission_cost = abs(current_price * size_asset * commission_pct)
                        if balance >= entry_commission_cost:
                            if self.order_executor:
                                self.order_executor.execute_order(self.symbol, entry_side, size_asset, current_price, 'limit')

                            sl = self.risk_manager.calculate_stop_loss(current_price, current_atr, side=entry_side) if self.risk_manager else None
                            tp = self.risk_manager.calculate_take_profit(current_price, current_atr, side=entry_side) if self.risk_manager else None
                            self.current_live_position = {
                                "side": entry_side, "entry_price": current_price, "size": size_asset,
                                "timestamp": current_timestamp, "sl_price": sl, "tp_price": tp,
                                "entry_commission": entry_commission_cost
                            }
                            balance -= entry_commission_cost # Deduct commission from balance upon entry
                            executed_action = f"ENTER_{entry_side.upper()}"
                            log_details.update({
                                "entry_price": current_price, "size": size_asset,
                                "commission_entry": entry_commission_cost,
                                "sl_price": sl, "tp_price": tp,
                                "balance_after_commission": balance
                            })
                        else:
                            self._log_simulation_action(current_timestamp, "SKIP_ENTRY_COMMISSION", {"balance": balance, "commission": entry_commission_cost})
                    else:
                        self._log_simulation_action(current_timestamp, "SKIP_ENTRY_SIZE", {"size_asset": size_asset})

                pos_after = self.current_live_position
                unrealized_pnl_after = 0
                if pos_after["side"] == "long": unrealized_pnl_after = (current_price - pos_after["entry_price"]) * pos_after["size"]
                elif pos_after["side"] == "short": unrealized_pnl_after = (pos_after["entry_price"] - current_price) * pos_after["size"]
                log_details["equity_after_action"] = float(balance + unrealized_pnl_after)

                self._log_simulation_action(current_timestamp, executed_action, log_details)

            except ccxt.NetworkError as e:
                print(f"[{now_utc.strftime('%H:%M:%S')}] SIM Network Error: {e}. Retrying or waiting...")
                self._log_simulation_action(now_utc, "ERROR_NETWORK", {"message": str(e)})
                time.sleep(min(5, loop_interval_seconds / 2))
            except Exception as e:
                print(f"[{now_utc.strftime('%H:%M:%S')}] SIM Loop Error: {e}")
                traceback.print_exc(limit=1)
                self._log_simulation_action(now_utc, "ERROR_LOOP", {"message": str(e), "traceback": traceback.format_exc(limit=1)})

            time.sleep(max(0.1, loop_interval_seconds - (time.monotonic() - loop_start_time)))

        print("--- Live Simulation Loop Finishing ---")
        if self.current_live_position["side"]:
            print("Closing open position at end of simulation...")
            df_final_ohlcv = self.data_handler.fetch_ohlcv(limit=1) if self.data_handler else None
            if df_final_ohlcv is not None and not df_final_ohlcv.empty:
                final_price = df_final_ohlcv['close'].iloc[-1]
                final_timestamp = pd.Timestamp(df_final_ohlcv['timestamp'].iloc[-1])
                pos_end = self.current_live_position

                if self.order_executor:
                    self.order_executor.execute_order(
                        self.symbol, "sell" if pos_end["side"] == "long" else "buy",
                        pos_end["size"], final_price, 'market'
                    )

                pnl_asset_end = (final_price - pos_end["entry_price"]) if pos_end["side"] == "long" else (pos_end["entry_price"] - final_price)
                pnl_gross_end = pnl_asset_end * pos_end["size"]
                commission_exit_cost_end = abs(final_price * pos_end["size"] * commission_pct)
                pnl_net_end = pnl_gross_end - pos_end.get("entry_commission", 0) - commission_exit_cost_end
                balance += pnl_net_end

                self.live_equity_history.append({'timestamp': final_timestamp, 'equity': balance})
                self._log_simulation_action(final_timestamp, f"EXIT_{pos_end['side'].upper()}_END_SIM", {
                    "price": final_price, "pnl_net_trade": pnl_net_end, "final_balance": balance
                })
                print(f"Closed {pos_end['side']} @{final_price:.4f}, PnL Net:{pnl_net_end:.4f}, Final Bal:{balance:.2f}")
            else:
                print("Warning (LiveSimulator): Could not fetch final price to close position.")
                self._log_simulation_action(datetime.now(timezone.utc), "ERROR_CLOSE_END_SIM", {"message": "Failed to fetch final price"})
                if self.live_equity_history and self.live_equity_history[-1]['equity'] is not None:
                    self.live_equity_history.append({'timestamp': datetime.now(timezone.utc), 'equity': self.live_equity_history[-1]['equity']})
                else: # Fallback if history is empty or last equity is None
                    self.live_equity_history.append({'timestamp': datetime.now(timezone.utc), 'equity': balance})

        self.simulation_running = False
        print("--- Live Simulation Thread Finished ---")

    def start_live_simulation(self, initial_equity=None, threshold=None, fetch_limit=None,
                              commission_pct=None, leverage=None):
        """Starts the live simulation thread."""
        if self.simulation_running:
            print("Simulation already running.")
            return
        if not self.exchange:
            print("Error (LiveSimulator): Exchange not initialized. Cannot start simulation.")
            return
        if not all([self.data_handler, self.feature_engineer, self.model_predictor, self.risk_manager, self.order_executor]):
            print("Error (LiveSimulator): One or more core components not initialized.")
            return

        _initial_equity = initial_equity if initial_equity is not None else self.config.get('initial_balance', 10000)
        _threshold = threshold if threshold is not None else self.config.get('simulation_threshold',
                                                                           self.config.get('backtest_threshold', 0.5))
        _fetch_limit = fetch_limit if fetch_limit is not None else self.config.get('fetch_live_limit', 300)
        _commission_pct = commission_pct if commission_pct is not None else self.config.get('commission_pct', 0.0006)
        _leverage = leverage if leverage is not None else self.config.get('leverage', 1)

        try:
            loop_interval = self.exchange.parse_timeframe(self.timeframe) if hasattr(self.exchange, 'parse_timeframe') else 60
            min_interval = self.config.get('min_simulation_interval_seconds', 15)
            loop_interval = max(min_interval, loop_interval)
        except Exception as e:
            print(f"Error (LiveSimulator) parsing timeframe '{self.timeframe}'. Using default 60s. Error: {e}")
            loop_interval = 60

        self.simulation_running = True
        self.simulation_stop_event.clear()
        self.live_equity_history = []

        self.simulation_thread = threading.Thread(
            target=self._simulation_loop,
            kwargs={
                'initial_equity': _initial_equity, 'threshold': _threshold,
                'fetch_limit': _fetch_limit, 'loop_interval_seconds': loop_interval,
                'commission_pct': _commission_pct, 'leverage': _leverage
            },
            daemon=True
        )
        self.simulation_thread.start()
        print(f"Live simulation thread started (Interval: {loop_interval}s). Log: {self.simulation_log_path}")

    def stop_live_simulation(self, wait_time=15):
        """Stops the running live simulation thread."""
        if not self.simulation_running or self.simulation_thread is None:
            print("Simulation not running or thread not found.")
            return

        print("Attempting to stop live simulation gracefully...")
        self.simulation_stop_event.set()
        if self.simulation_thread.is_alive(): # Only join if alive
            self.simulation_thread.join(timeout=wait_time)

        if self.simulation_thread.is_alive():
            print(f"Warning (LiveSimulator): Simulation thread did not stop within {wait_time}s.")
        else:
            print("Simulation thread stopped.")

        self.simulation_running = False
        self.simulation_thread = None

        print("Live simulation stopped.")

    def get_simulation_equity_data(self):
        """Returns the collected equity history for plotting."""
        return pd.DataFrame(self.live_equity_history) if self.live_equity_history else pd.DataFrame()