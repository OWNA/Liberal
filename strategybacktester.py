# strategy_backtester.py
# Reformatted from notebook export to standard Python file

import os
import pandas as pd
import traceback

# pandas_ta might be needed if ATR is recalculated here,
# but ideally, it's pre-calculated and passed in df_with_signals.
# For this version, we assume 'atr' is present or a fallback is used.

class StrategyBacktester:
    """
    Performs backtesting of a trading strategy.
    """
    def __init__(self, config, risk_manager, data_handler=None, feature_engineer=None,
                 has_pandas_ta=False, ta_module=None):
        """
        Initializes the StrategyBacktester.

        Args:
            config (dict): Configuration dictionary.
            risk_manager (AdvancedRiskManager): Instance of AdvancedRiskManager.
            data_handler (DataHandler, optional): Instance of DataHandler for fetching/
                re-processing if needed (e.g. for ATR).
            feature_engineer (FeatureEngineer, optional): Instance of FeatureEngineer for
                re-calculating features if needed.
            has_pandas_ta (bool): Flag indicating if pandas_ta is available.
            ta_module: The imported pandas_ta module.
        """
        self.config = config
        self.risk_manager = risk_manager
        self.data_handler = data_handler
        self.feature_engineer = feature_engineer
        self.HAS_PANDAS_TA = has_pandas_ta
        self.ta = ta_module

        self.initial_balance = config.get('initial_balance', 10000)
        self.commission_pct = config.get('commission_pct', 0.0006)
        self.leverage = config.get('leverage', 1)

        self.base_dir = config.get('base_dir', './trading_bot_data')
        safe_symbol = config.get('symbol', 'SYMBOL').replace('/', '_').replace(':', '')
        timeframe = config.get('timeframe', 'TIMEFRAME')
        self.backtest_log_path = os.path.join(
            self.base_dir,
            f"backtest_log_{safe_symbol}_{timeframe}.csv"
        )

        print("StrategyBacktester initialized.")

    def _ensure_atr(self, df_input):
        """
        Ensures the 'atr' column is present in the DataFrame.
        If not, and if DataHandler/FeatureEngineer/pandas_ta are available, tries to calculate it.
        Otherwise, uses a fallback.
        """
        df = df_input.copy()
        if 'atr' in df.columns and df['atr'].notna().all():
            return df

        print("Warning (Backtester): 'atr' column missing or has NaNs. Attempting to handle.")

        # Attempt to recalculate ATR if necessary components are available
        if self.HAS_PANDAS_TA and self.ta:
            required_cols_for_atr = ['high', 'low', 'close']
            if all(col in df.columns for col in required_cols_for_atr):
                atr_lookback = self.config.get('risk_management', {}).get('volatility_lookback', 14)
                try:
                    print(f"Recalculating ATR with lookback {atr_lookback} using pandas_ta.")
                    df['atr_recalc'] = self.ta.atr(df['high'], df['low'], df['close'], length=atr_lookback)
                    # If original 'atr' column exists, fill its NaNs; otherwise, use the recalculated one
                    if 'atr' in df.columns:
                        df['atr'] = df['atr'].fillna(df['atr_recalc'])
                    else:
                        df['atr'] = df['atr_recalc']
                    df.drop(columns=['atr_recalc'], inplace=True, errors='ignore')

                    # Drop rows where ATR could still not be calculated (e.g., at the very beginning)
                    initial_len = len(df)
                    df.dropna(subset=['atr'], inplace=True)
                    if len(df) < initial_len:
                        print(f"Dropped {initial_len - len(df)} rows due to NaN ATR after recalculation.")

                    if not df.empty and df['atr'].notna().all():
                        print("ATR successfully recalculated/filled.")
                        return df
                    else:
                        print("ATR recalculation/fill did not resolve all NaNs.")
                except Exception as e:
                    print(f"Error (Backtester): Failed to recalculate ATR using pandas_ta: {e}.")
            else:
                print("Warning (Backtester): Cannot recalculate ATR, missing high, low, or close columns.")
        else:
            print("Warning (Backtester): pandas_ta not available. Cannot recalculate ATR.")

        # Fallback if ATR is still NaN or could not be calculated
        if 'atr' not in df.columns or df['atr'].isnull().any():
            fallback_atr_pct = self.config.get('fallback_atr_pct_for_backtest', 0.02) # 2% of close
            print(f"Using fallback ATR: {fallback_atr_pct*100}% of close price.")
            if 'close' in df.columns:
                if 'atr' in df.columns:
                    df['atr'] = df['atr'].fillna(fallback_atr_pct * df['close'])
                else:
                    df['atr'] = fallback_atr_pct * df['close']
                df.dropna(subset=['atr'], inplace=True) # Drop if close was NaN leading to NaN ATR
            else:
                print("Error (Backtester): 'close' column also missing. Cannot create fallback ATR.")
                return pd.DataFrame()

        if df.empty:
            print("Error (Backtester): DataFrame empty after ensuring ATR. Cannot proceed.")
        return df

    def run_backtest(self, df_with_signals):
        """
        Performs the backtest simulation.
        """
        if df_with_signals is None or df_with_signals.empty:
            print("Error (Backtester): Input DataFrame for backtest is empty.")
            return None, None

        if 'signal' not in df_with_signals.columns or 'close' not in df_with_signals.columns:
            print("Error (Backtester): DataFrame must contain 'signal' and 'close' columns.")
            return None, None

        backtest_df = self._ensure_atr(df_with_signals.copy())
        if backtest_df.empty:
            print("Error (Backtester): DataFrame empty after ensuring ATR. Cannot proceed with backtest.")
            return None, None

        print(f"Running backtest on {len(backtest_df)} data points...")

        balance = self.initial_balance
        position = 0
        entry_price = 0.0
        size_asset = 0.0
        stop_loss_price = None
        take_profit_price = None
        entry_timestamp = None
        entry_commission_cost = 0.0

        equity_curve_data = [] # Changed from equity_curve to avoid confusion with df column name
        trades = []

        for row_idx, row in enumerate(backtest_df.itertuples(index=False)): # index=False for namedtuple access
            current_price = row.close
            current_signal = row.signal
            current_timestamp = row.timestamp
            current_high = row.high
            current_low = row.low
            current_atr = row.atr

            unrealized_pnl = 0
            if position == 1: unrealized_pnl = (current_price - entry_price) * size_asset
            elif position == -1: unrealized_pnl = (entry_price - current_price) * size_asset

            current_equity = balance + unrealized_pnl
            equity_curve_data.append({'timestamp': current_timestamp, 'equity': current_equity})

            exit_reason = None
            exit_price = current_price

            if position == 1:
                if stop_loss_price and current_low <= stop_loss_price:
                    exit_reason, exit_price, current_signal = "SL_HIT", stop_loss_price, -1
                elif take_profit_price and current_high >= take_profit_price:
                    exit_reason, exit_price, current_signal = "TP_HIT", take_profit_price, -1
            elif position == -1:
                if stop_loss_price and current_high >= stop_loss_price:
                    exit_reason, exit_price, current_signal = "SL_HIT", stop_loss_price, 1
                elif take_profit_price and current_low <= take_profit_price:
                    exit_reason, exit_price, current_signal = "TP_HIT", take_profit_price, 1

            if position != 0 and (current_signal == -position or (current_signal == 0 and position != 0) or exit_reason):
                pnl_per_asset = (exit_price - entry_price) if position == 1 else (entry_price - exit_price)
                pnl_gross = pnl_per_asset * size_asset
                commission_exit_cost = abs(exit_price * size_asset * self.commission_pct)
                pnl_net = pnl_gross - entry_commission_cost - commission_exit_cost
                balance += pnl_net

                trades.append({
                    "entry_timestamp": entry_timestamp, "exit_timestamp": current_timestamp,
                    "direction": "long" if position == 1 else "short",
                    "entry_price": entry_price, "exit_price": exit_price,
                    "size_asset": size_asset, "pnl_net": pnl_net,
                    "commission_total": entry_commission_cost + commission_exit_cost,
                    "equity_after_trade": balance,
                    "exit_reason": exit_reason or "SignalFlipOrNeutral"
                })
                position, entry_price, size_asset = 0, 0.0, 0.0
                stop_loss_price, take_profit_price = None, None
                entry_timestamp, entry_commission_cost = None, 0.0

            if position == 0 and current_signal != 0 and not exit_reason:
                position = current_signal
                entry_price = current_price
                entry_timestamp = current_timestamp

                volatility_pct = (current_atr / current_price) if current_price > 0 and pd.notna(current_atr) and current_atr > 1e-9 else \
                                 self.config.get('fallback_volatility_pct_for_sizing', 0.02)

                target_position_size_usd = self.risk_manager.calculate_position_size(current_equity, volatility_pct) # Use current_equity before this trade
                size_asset = (target_position_size_usd * self.leverage) / current_price if current_price > 0 else 0

                if size_asset <= 1e-8:
                    position = 0
                    continue

                entry_commission_cost = abs(entry_price * size_asset * self.commission_pct)
                # Balance is not reduced by commission here; it's accounted for in PnL net.

                sl_tp_side = "long" if position == 1 else "short"
                stop_loss_price = self.risk_manager.calculate_stop_loss(entry_price, current_atr, side=sl_tp_side)
                take_profit_price = self.risk_manager.calculate_take_profit(entry_price, current_atr, side=sl_tp_side)

        if position != 0: # Close open position at the end
            last_price = backtest_df['close'].iloc[-1]
            last_timestamp = backtest_df['timestamp'].iloc[-1]

            pnl_per_asset = (last_price - entry_price) if position == 1 else (entry_price - last_price)
            pnl_gross = pnl_per_asset * size_asset
            commission_exit_cost = abs(last_price * size_asset * self.commission_pct)
            pnl_net = pnl_gross - entry_commission_cost - commission_exit_cost
            balance += pnl_net

            if equity_curve_data: equity_curve_data[-1]['equity'] = balance

            trades.append({
                "entry_timestamp": entry_timestamp, "exit_timestamp": last_timestamp,
                "direction": "long" if position == 1 else "short",
                "entry_price": entry_price, "exit_price": last_price,
                "size_asset": size_asset, "pnl_net": pnl_net,
                "commission_total": entry_commission_cost + commission_exit_cost,
                "equity_after_trade": balance, "exit_reason": "EndOfBacktest"
            })

        if not equity_curve_data:
            print("Warning (Backtester): No trades or equity points generated.")
            # Create a dummy equity curve if no trades
            equity_df_final = backtest_df[['timestamp']].copy()
            equity_df_final['equity'] = self.initial_balance
            return equity_df_final, pd.DataFrame(trades)

        results_df_equity_only = pd.DataFrame(equity_curve_data)
        final_results_df = pd.merge(backtest_df, results_df_equity_only, on='timestamp', how='left')
        final_results_df['equity'].ffill(inplace=True)

        first_valid_equity_idx = final_results_df['equity'].first_valid_index()
        if first_valid_equity_idx is not None:
            final_results_df['equity'].iloc[:first_valid_equity_idx] = self.initial_balance
        elif not final_results_df.empty : # If all equity is NaN (e.g. only one row of data)
            final_results_df['equity'] = self.initial_balance

        trades_log_df = pd.DataFrame(trades)

        if not trades_log_df.empty: self.save_backtest_log(trades_log_df)

        print(f"Backtest complete. Final balance: {balance:.2f}")
        return final_results_df, trades_log_df

    def save_backtest_log(self, trades_log_df):
        """Saves the detailed trade log from a backtest to a CSV file."""
        try:
            os.makedirs(os.path.dirname(self.backtest_log_path), exist_ok=True)
            trades_log_df.to_csv(self.backtest_log_path, index=False, float_format='%.8f')
            print(f"Backtest trade log saved to {self.backtest_log_path}")
        except Exception as e:
            print(f"Warning (Backtester): Error saving backtest log: {e}")
            traceback.print_exc()