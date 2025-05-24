# trading_bot_orchestrator.py
# Reformatted from notebook export to standard Python file

import os
import time
import pandas as pd
import numpy as np  # For np.nan
import ccxt  # For exchange initialization and potential errors
import traceback  # For detailed error logging
from sklearn.model_selection import KFold  # For walk-forward, though manual slicing is often used for time series

# Import all refactored classes
from advancedriskmanager import AdvancedRiskManager
from smartorderexecutor import SmartOrderExecutor
from datahandler import DataHandler
from featureengineer import FeatureEngineer
from labelgenerator import LabelGenerator
from modeltrainer import ModelTrainer
from modelpredictor import ModelPredictor
from strategybacktester import StrategyBacktester
from livesimulator import LiveSimulator
from visualizer import Visualizer


class TradingBotOrchestrator:
    """
    Orchestrates the workflow of the trading bot using refactored components.
    (Phase 1 Update: Includes Walk-Forward Optimization structure)
    """

    def __init__(self, config, api_key=None, api_secret=None,
                 global_library_flags=None, global_library_modules=None):
        """
        Initializes the TradingBotOrchestrator and its components.
        (Constructor remains largely the same)
        """
        self.config = config
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange = None

        self.lib_flags = global_library_flags or {}
        self.lib_modules = global_library_modules or {}

        self.df_historical_data = pd.DataFrame()
        self.df_features = pd.DataFrame()
        self.df_labeled_features = pd.DataFrame()
        self.trained_model_booster = None
        self.trained_ensemble_models = None
        self.trained_features_list = []
        self.target_mean_for_prediction = None
        self.target_std_for_prediction = None
        self.backtest_results = pd.DataFrame()
        self.backtest_trades_log = pd.DataFrame()
        self.walk_forward_results_summary = []  # For storing results from each WFO fold

        self._initialize_exchange()
        self._initialize_components()

        print("TradingBotOrchestrator initialized (Phase 1 Update).")

    def _initialize_exchange(self):
        # (No changes from previous version)
        ccxt_module = self.lib_modules.get('ccxt')
        if not ccxt_module:
            print("FATAL (Orchestrator): CCXT library module not available. Cannot initialize exchange.")
            return
        try:
            exchange_config = {'enableRateLimit': True, 'options': {'adjustForTimeDifference': True}}
            if self.api_key and self.api_secret:
                exchange_config.update({'apiKey': self.api_key, 'secret': self.api_secret})

            exchange_name = self.config.get('exchange_name', 'bybit')
            if not hasattr(ccxt_module, exchange_name):
                print(f"FATAL (Orchestrator): CCXT does not support exchange '{exchange_name}'.")
                return

            self.exchange = getattr(ccxt_module, exchange_name)(exchange_config)
            if self.config.get('exchange_testnet', False):
                self.exchange.set_sandbox_mode(True)
                print(f"Exchange '{exchange_name}' initialized in SANDBOX/TESTNET mode.")
            else:
                print(f"Exchange '{exchange_name}' initialized in LIVE mode.")

            self.exchange.load_markets()
            print(f"Markets loaded for '{exchange_name}'.")
        except Exception as e:
            print(f"FATAL (Orchestrator): CCXT Exchange Error initializing '{self.config.get('exchange_name', 'bybit')}': {e}")
            traceback.print_exc()
            self.exchange = None

    def _initialize_components(self):
        # (No changes from previous version in terms of component instantiation,
        # but the components themselves are the updated Phase 1 versions)
        if not self.exchange and not self.config.get('allow_no_exchange_init', False):
            print("Warning (Orchestrator): Exchange not initialized. Some components may not function.")

        self.risk_manager = AdvancedRiskManager(self.config.get('risk_management', {})) if AdvancedRiskManager else None
        if SmartOrderExecutor:
            if self.exchange:
                self.order_executor = SmartOrderExecutor(self.exchange, self.config.get('execution', {}))
            elif self.config.get('allow_no_exchange_init', False):
                self.order_executor = None
                print("Warning (Orchestrator): SmartOrderExecutor not initialized as exchange is not available (but allowed).")
            else:
                self.order_executor = None
        else:
            self.order_executor = None

        if DataHandler:
            if self.exchange or self.config.get('allow_no_exchange_init', False):
                self.data_handler = DataHandler(self.config, self.exchange)
            else:
                self.data_handler = None
        else:
            self.data_handler = None

        self.feature_engineer = FeatureEngineer(
            config=self.config,
            has_pandas_ta=self.lib_flags.get('HAS_PANDAS_TA', False),
            has_pyemd=self.lib_flags.get('HAS_PYEMD', False),
            has_scipy_hilbert=self.lib_flags.get('HAS_SCIPY_HILBERT', False),
            ta_module=self.lib_modules.get('ta'),
            emd_class=self.lib_modules.get('EMD'),
            hilbert_func=self.lib_modules.get('hilbert')
        ) if FeatureEngineer else None

        self.label_generator = LabelGenerator(self.config) if LabelGenerator else None

        self.model_trainer = ModelTrainer(
            config=self.config,
            feature_list_all_defined=self.feature_engineer.all_defined_feature_columns if self.feature_engineer else [],
            has_optuna=self.lib_flags.get('HAS_OPTUNA', False),
            optuna_module=self.lib_modules.get('optuna')
        ) if ModelTrainer else None

        self.model_predictor = ModelPredictor(
            config=self.config,
            data_handler=self.data_handler,
            label_generator=self.label_generator
        ) if ModelPredictor else None

        self.backtester = StrategyBacktester(
            config=self.config,
            risk_manager=self.risk_manager,
            data_handler=self.data_handler,
            feature_engineer=self.feature_engineer,
            has_pandas_ta=self.lib_flags.get('HAS_PANDAS_TA', False),
            ta_module=self.lib_modules.get('ta')
        ) if StrategyBacktester and self.risk_manager else None

        if LiveSimulator and self.exchange and self.data_handler and self.feature_engineer and \
           self.model_predictor and self.risk_manager and self.order_executor:
            self.live_simulator = LiveSimulator(
                config=self.config, exchange_api=self.exchange,
                data_handler=self.data_handler, feature_engineer=self.feature_engineer,
                model_predictor=self.model_predictor, risk_manager=self.risk_manager,
                order_executor=self.order_executor
            )
        else:
            self.live_simulator = None

        self.visualizer = Visualizer(
            config=self.config,
            has_matplotlib=self.lib_flags.get('HAS_MATPLOTLIB', False), plt_module=self.lib_modules.get('plt'),
            has_shap=self.lib_flags.get('HAS_SHAP', False), shap_module=self.lib_modules.get('shap'),
            has_pyemd=self.lib_flags.get('HAS_PYEMD', False), emd_class=self.lib_modules.get('EMD'),
            has_scipy_hilbert=self.lib_flags.get('HAS_SCIPY_HILBERT', False), hilbert_func=self.lib_modules.get('hilbert')
        ) if Visualizer else None

        # (Warnings for uninitialized components remain the same)
        if not self.data_handler:
            print("Warning (Orchestrator Init): DataHandler failed to initialize.")
        # ... (other checks) ...
        if not self.visualizer:
            print("Warning (Orchestrator Init): Visualizer failed to initialize.")

    def prepare_data_for_training(self, df_input=None, save_features=True, save_ohlcv=True):
        """
        Loads/cleans OHLCV, aligns L2 (if configured), engineers features, and generates labels.
        Can optionally take a DataFrame as input (e.g., a slice for walk-forward).
        """
        if not self.data_handler or not self.feature_engineer or not self.label_generator:
            print("Error (Orchestrator): Data processing components not initialized.")
            return False

        print("\n--- Starting Data Preparation ---")

        current_df_historical = pd.DataFrame()
        if df_input is not None and not df_input.empty:
            print("Using provided DataFrame as input for data preparation.")
            # Assume df_input is raw OHLCV data for this fold/slice
            # Clean it first
            current_df_historical = self.data_handler.clean_ohlcv_data(df_input)
            if current_df_historical.empty:
                print("Error (Orchestrator): Provided input DataFrame became empty after cleaning.")
                return False
            # Note: L2 alignment for a slice in WFO needs careful handling of L2 data source.
            # For simplicity here, we assume L2 is either not used for the slice or handled if `use_historical_l2` is true.
        else:
            print("No input DataFrame provided, loading full historical data.")
            use_hist_l2 = self.config.get('use_l2_features_for_training', self.config.get('use_l2_features', False))
            current_df_historical = self.data_handler.load_and_prepare_historical_data(
                use_historical_l2=use_hist_l2,
                fetch_ohlcv_limit=self.config.get('fetch_ohlcv_limit'),
                save_ohlcv=save_ohlcv  # Control saving of full OHLCV
            )
        if current_df_historical.empty:
            print("Error (Orchestrator): Failed to load/prepare historical data.")
            self.df_historical_data = pd.DataFrame()  # Ensure it's empty
            return False

        self.df_historical_data = current_df_historical  # Store the data used for this prep step

        # Generate features on this specific (potentially sliced) data
        current_df_features = self.feature_engineer.generate_all_features(self.df_historical_data, save=save_features)
        if current_df_features.empty:
            print("Error (Orchestrator): Failed to engineer features.")
            self.df_features = pd.DataFrame()
            return False
        self.df_features = current_df_features

        current_df_labeled, mean_val, std_val = self.label_generator.generate_labels(self.df_features)
        if current_df_labeled.empty:
            print("Error (Orchestrator): Failed to generate labels.")
            self.df_labeled_features = pd.DataFrame()
            return False
        self.df_labeled_features = current_df_labeled

        # Store scaling parameters from this specific training data preparation
        # These might change per fold in walk-forward if labels are re-generated per fold
        self.target_mean_for_prediction = mean_val
        self.target_std_for_prediction = std_val
        if self.model_predictor:
            self.model_predictor.set_scaling_params(mean_val, std_val)

        print("Data preparation complete for current data segment.")
        return True

    def train_model(self, df_training_data=None, save_model=True):
        """
        Trains the model (standard or ensemble based on config) on provided data.
        If df_training_data is None, uses self.df_labeled_features.
        """
        data_to_train_on = df_training_data if df_training_data is not None and not df_training_data.empty else self.df_labeled_features

        if not self.model_trainer or data_to_train_on is None or data_to_train_on.empty:
            print("Error (Orchestrator): ModelTrainer not initialized or no labeled data available for training.")
            return False

        print("\n--- Starting Model Training ---")
        # --- Phase 1c: Pass feature selection config to ModelTrainer (handled internally by ModelTrainer now via config) ---
        # The ModelTrainer's _prepare_training_data already handles feature selection placeholder

        train_ensemble = self.config.get('train_ensemble', False)

        # Pass current scaling parameters to config so ModelTrainer can save them if needed
        # (especially for standard model's feature JSON, ensemble saves it in pickle)
        self.config['target_mean_for_prediction'] = self.target_mean_for_prediction
        self.config['target_std_for_prediction'] = self.target_std_for_prediction

        if train_ensemble:
            print("Training ensemble model...")
            ensemble_models, features_used = self.model_trainer.train_ensemble_model(data_to_train_on, save=save_model)
            if ensemble_models:
                self.trained_ensemble_models = ensemble_models  # Store the latest trained ensemble
                self.trained_features_list = features_used
                print("Ensemble model training successful.")
                return True
        else:
            print("Training standard model...")
            booster, features_used = self.model_trainer.train_standard_model(data_to_train_on, save=save_model)
            if booster:
                self.trained_model_booster = booster  # Store the latest trained booster
                self.trained_features_list = features_used
                print("Standard model training successful.")
                return True

        print("Error (Orchestrator): Model training failed.")
        return False

    def run_backtest(self, df_backtest_data=None, load_latest_model=True):
        """
        Runs the backtesting process on provided data.
        If df_backtest_data is None, uses self.df_features.
        If load_latest_model is True, loads the model specified by config (ensemble or standard).
        """
        if not self.backtester or not self.model_predictor:
            print("Error (Orchestrator): Backtester or ModelPredictor not initialized.")
            return pd.DataFrame(), pd.DataFrame()  # Return empty DFs on failure

        data_for_backtest = df_backtest_data if df_backtest_data is not None and not df_backtest_data.empty else self.df_features

        if data_for_backtest.empty:
            print("Error (Orchestrator): No feature data available for backtesting.")
            # Attempt to load from prepared_data.csv as a last resort
            if hasattr(self.feature_engineer, 'prepared_data_path') and \
               os.path.exists(self.feature_engineer.prepared_data_path):
                print("Attempting to load from prepared_data.csv for backtest.")
                try:
                    data_for_backtest = pd.read_csv(self.feature_engineer.prepared_data_path, parse_dates=['timestamp'])
                    if 'timestamp' in data_for_backtest.columns and data_for_backtest['timestamp'].dt.tz is None:
                        data_for_backtest['timestamp'] = data_for_backtest['timestamp'].dt.tz_localize('utc')
                    elif 'timestamp' in data_for_backtest.columns:
                        data_for_backtest['timestamp'] = data_for_backtest['timestamp'].dt.tz_convert('utc')
                except Exception as e:
                    print(f"Error loading prepared_data.csv for backtest: {e}")
                    return pd.DataFrame(), pd.DataFrame()
            if data_for_backtest.empty:
                return pd.DataFrame(), pd.DataFrame()

        print("\n--- Starting Backtest ---")
        if load_latest_model:
            use_ensemble_for_pred = self.config.get('use_ensemble_for_backtest', self.config.get('train_ensemble', False))
            if not self.model_predictor.load_model_and_features(load_ensemble=use_ensemble_for_pred):
                print("Error (Orchestrator): Failed to load model for backtest predictions.")
                return pd.DataFrame(), pd.DataFrame()

        # Ensure scaling params are set in predictor (should be from last training or loaded with model)
        if self.model_predictor.target_mean is None or self.model_predictor.target_std is None:
            print("Warning (Orchestrator): Scaling params not set in ModelPredictor for backtest. Attempting to use orchestrator's last known values or recalculate.")
            if self.target_mean_for_prediction is not None and self.target_std_for_prediction is not None:
                self.model_predictor.set_scaling_params(self.target_mean_for_prediction, self.target_std_for_prediction)
            elif not self.model_predictor._ensure_scaling_info():
                print("Critical Warning (Orchestrator): Could not ensure scaling info for backtest. Predictions may be affected.")

        df_with_signals = self.model_predictor.predict_signals(
            data_for_backtest,
            use_ensemble=self.config.get('use_ensemble_for_backtest', self.config.get('train_ensemble', False))
        )
        if df_with_signals is None or df_with_signals.empty:
            print("Error (Orchestrator): Failed to generate signals for backtest.")
            return pd.DataFrame(), pd.DataFrame()

        # Store results in instance variables
        self.backtest_results, self.backtest_trades_log = self.backtester.run_backtest(df_with_signals)

        if self.backtest_results is not None and not self.backtest_results.empty:
            print("Backtest complete.")
            if self.visualizer:
                initial_bal = self.backtester.initial_balance
                final_bal = self.backtest_results['equity'].iloc[-1] if not self.backtest_results['equity'].empty else initial_bal
                return_pct = ((final_bal - initial_bal) / initial_bal * 100) if initial_bal != 0 else 0
                self.visualizer.plot_equity_curve(self.backtest_results, initial_bal, final_bal, return_pct, "Backtest")
            return self.backtest_results, self.backtest_trades_log
        else:
            print("Error (Orchestrator): Backtest failed to produce results or results DataFrame is empty.")
            return pd.DataFrame(), pd.DataFrame()

    def run_walk_forward_optimization(self):
        """
        Performs walk-forward optimization.
        Trains on a rolling/expanding window of past data and tests on a subsequent out-of-sample window.
        """
        print("\n--- Starting Walk-Forward Optimization ---")
        if not self.data_handler or not self.feature_engineer or not self.label_generator or \
           not self.model_trainer or not self.model_predictor or not self.backtester:
            print("Error (Orchestrator WFO): One or more core components are not initialized.")
            return False

        # WFO Parameters from config
        train_periods = self.config.get('walk_forward_train_periods', 250)  # Number of bars/candles
        test_periods = self.config.get('walk_forward_test_periods', 60)
        step_periods = self.config.get('walk_forward_step_periods', test_periods)  # How much to slide window forward
        initial_warmup_periods = self.config.get('walk_forward_initial_warmup', 50)  # For initial feature calculation stability
        retrain_frequency_folds = self.config.get('walk_forward_retrain_frequency_folds', 1)  # Retrain model every N test folds

        # Load full dataset once
        print("Loading full historical data for WFO...")
        full_ohlcv_data = self.data_handler.fetch_ohlcv(limit=self.config.get('fetch_ohlcv_limit_wfo', 5000))  # Fetch a large dataset
        if full_ohlcv_data.empty or len(full_ohlcv_data) < (train_periods + test_periods + initial_warmup_periods):
            print(f"Error (Orchestrator WFO): Insufficient data for walk-forward. Need at least {train_periods + test_periods + initial_warmup_periods} periods, got {len(full_ohlcv_data)}.")
            return False

        full_ohlcv_data = self.data_handler.clean_ohlcv_data(full_ohlcv_data)
        if full_ohlcv_data.empty:
            return False

        all_fold_trades = []
        all_fold_equity_curves = []  # To store equity curves from each test fold
        self.walk_forward_results_summary = []  # Reset summary

        start_index = 0  # initial_warmup_periods # Start after warmup if features need it
        fold_number = 0

        while start_index + train_periods + test_periods <= len(full_ohlcv_data):
            fold_number += 1
            print(f"\n--- WFO Fold {fold_number} ---")

            # Define data slices for this fold
            # Training data needs enough history for feature calculation lookbacks
            train_slice_start = start_index
            train_slice_end = start_index + train_periods
            test_slice_start = train_slice_end
            test_slice_end = test_slice_start + test_periods

            print(f"Train period: {full_ohlcv_data['timestamp'].iloc[train_slice_start].date()} to {full_ohlcv_data['timestamp'].iloc[train_slice_end-1].date()} ({train_periods} bars)")
            print(f"Test period:  {full_ohlcv_data['timestamp'].iloc[test_slice_start].date()} to {full_ohlcv_data['timestamp'].iloc[test_slice_end-1].date()} ({test_periods} bars)")

            df_train_fold_ohlcv = full_ohlcv_data.iloc[train_slice_start:train_slice_end].copy()
            df_test_fold_ohlcv_raw = full_ohlcv_data.iloc[test_slice_start:test_slice_end].copy()  # Raw OHLCV for testing

            # --- Data Prep & Training for the current training fold ---
            # Pass df_train_fold_ohlcv to prepare_data_for_training
            # save_features=False, save_ohlcv=False as these are intermediate
            if not self.prepare_data_for_training(df_input=df_train_fold_ohlcv, save_features=False, save_ohlcv=False):
                print(f"WFO Fold {fold_number}: Data preparation for training failed. Skipping fold.")
                start_index += step_periods
                continue

            # Train model only if it's a retrain fold
            if fold_number == 1 or (fold_number - 1) % retrain_frequency_folds == 0:
                print(f"WFO Fold {fold_number}: Retraining model...")
                if not self.train_model(df_training_data=self.df_labeled_features, save_model=False):  # Don't save intermediate models
                    print(f"WFO Fold {fold_number}: Model training failed. Skipping fold.")
                    start_index += step_periods
                    continue
            else:
                print(f"WFO Fold {fold_number}: Using previously trained model.")
                if not (self.trained_model_booster or self.trained_ensemble_models):
                    print("Error (Orchestrator WFO): No model available from previous fold. This shouldn't happen if retrain_frequency is handled correctly.")
                    start_index += step_periods
                    continue

            # --- Prepare Test Data (Features for the test fold) ---
            # We need to generate features for the test data.
            # Crucially, feature engineering on test data should NOT use future information from the test period itself for rolling calculations.
            # One common way: concatenate train_ohlcv + test_ohlcv_raw, generate features on combined, then slice out test features.
            # This ensures rolling features at start of test period use training data.

            # For L2 features on test data, if historical L2 is used, it needs careful alignment for the test period.
            # If live-like L2 fetching is simulated, that's different.
            # For simplicity here, assume L2 features are either not used or `generate_all_features` handles it based on available `l2_bids`/`l2_asks` if already merged.

            # We need a segment of data that includes the training data + test data for continuous feature calculation
            # The lookback for features is self.feature_engineer.feature_window
            # So, take `feature_window` bars from end of train_fold_ohlcv and concat with test_fold_ohlcv_raw

            # More robust: Use a combined segment for feature engineering
            # Ensure enough history for feature calculation lookbacks before the test period starts
            feature_calc_start_idx = max(0, test_slice_start - (self.feature_engineer.feature_window + initial_warmup_periods))
            df_for_test_feature_gen_ohlcv = full_ohlcv_data.iloc[feature_calc_start_idx:test_slice_end].copy()

            print(f"WFO Fold {fold_number}: Generating features for test data segment (length: {len(df_for_test_feature_gen_ohlcv)})...")
            df_test_fold_all_features = self.feature_engineer.generate_all_features(df_for_test_feature_gen_ohlcv, save=False)

            if df_test_fold_all_features.empty:
                print(f"WFO Fold {fold_number}: Feature generation for test data failed. Skipping fold.")
                start_index += step_periods
                continue

            # Slice out only the test period features
            # Align by timestamp to be robust
            test_start_timestamp = df_test_fold_ohlcv_raw['timestamp'].iloc[0]
            test_end_timestamp = df_test_fold_ohlcv_raw['timestamp'].iloc[-1]

            df_test_fold_features = df_test_fold_all_features[
                (df_test_fold_all_features['timestamp'] >= test_start_timestamp) &
                (df_test_fold_all_features['timestamp'] <= test_end_timestamp)
            ].copy()

            if df_test_fold_features.empty:
                print(f"WFO Fold {fold_number}: Test features DataFrame is empty after slicing. Skipping fold.")
                start_index += step_periods
                continue

            # --- Backtest on the current test fold ---
            # ModelPredictor already has the model loaded from the training step (or previous fold)
            # and scaling parameters from the last training data prep.
            print(f"WFO Fold {fold_number}: Running backtest on test data ({len(df_test_fold_features)} bars)...")
            fold_backtest_results, fold_trades_log = self.run_backtest(df_backtest_data=df_test_fold_features, load_latest_model=False)  # Use current model in predictor

            if fold_backtest_results is not None and not fold_backtest_results.empty:
                all_fold_trades.append(fold_trades_log)
                all_fold_equity_curves.append(fold_backtest_results[['timestamp', 'equity']])  # Store for combined plot

                # Calculate and store summary stats for this fold
                initial_bal_fold = fold_backtest_results['equity'].iloc[0] if not fold_backtest_results['equity'].empty else self.backtester.initial_balance
                final_bal_fold = fold_backtest_results['equity'].iloc[-1] if not fold_backtest_results['equity'].empty else initial_bal_fold
                return_pct_fold = ((final_bal_fold - initial_bal_fold) / initial_bal_fold * 100) if initial_bal_fold != 0 else 0
                num_trades_fold = len(fold_trades_log) if fold_trades_log is not None else 0

                fold_summary = {
                    'fold': fold_number,
                    'train_start': df_train_fold_ohlcv['timestamp'].iloc[0].date(),
                    'train_end': df_train_fold_ohlcv['timestamp'].iloc[-1].date(),
                    'test_start': df_test_fold_ohlcv_raw['timestamp'].iloc[0].date(),
                    'test_end': df_test_fold_ohlcv_raw['timestamp'].iloc[-1].date(),
                    'initial_equity': initial_bal_fold,
                    'final_equity': final_bal_fold,
                    'return_pct': return_pct_fold,
                    'num_trades': num_trades_fold,
                    'profit_factor': (
                        fold_trades_log[fold_trades_log['pnl_net'] > 0]['pnl_net'].sum() /
                        abs(fold_trades_log[fold_trades_log['pnl_net'] < 0]['pnl_net'].sum())
                    ) if num_trades_fold > 0 and abs(fold_trades_log[fold_trades_log['pnl_net'] < 0]['pnl_net'].sum()) > 1e-9 else np.nan,
                    'win_rate': (
                        len(fold_trades_log[fold_trades_log['pnl_net'] > 0]) / num_trades_fold * 100
                    ) if num_trades_fold > 0 else np.nan
                }
                self.walk_forward_results_summary.append(fold_summary)
                print(f"WFO Fold {fold_number} Summary: Return {return_pct_fold:.2f}%, Trades {num_trades_fold}")
            else:
                print(f"WFO Fold {fold_number}: Backtest on test data produced no results.")
                # Still add a summary for this failed fold
                self.walk_forward_results_summary.append({
                    'fold': fold_number,
                    'train_start': df_train_fold_ohlcv['timestamp'].iloc[0].date(),
                    'train_end': df_train_fold_ohlcv['timestamp'].iloc[-1].date(),
                    'test_start': df_test_fold_ohlcv_raw['timestamp'].iloc[0].date(),
                    'test_end': df_test_fold_ohlcv_raw['timestamp'].iloc[-1].date(),
                    'return_pct': np.nan,
                    'num_trades': 0
                })

            start_index += step_periods  # Slide window

        # --- Aggregate and Display WFO Results ---
        if not self.walk_forward_results_summary:
            print("Walk-Forward Optimization completed, but no fold results were generated.")
            return False

        summary_df = pd.DataFrame(self.walk_forward_results_summary)
        print("\n--- Walk-Forward Optimization Summary ---")
        print(summary_df.to_string())

        # Save summary to CSV
        summary_path = os.path.join(self.config.get('base_dir', '.'), "walk_forward_summary.csv")
        try:
            summary_df.to_csv(summary_path, index=False)
            print(f"WFO summary saved to {summary_path}")
        except Exception as e:
            print(f"Error saving WFO summary: {e}")

        # Concatenate all trade logs
        if all_fold_trades:
            combined_trades_log = pd.concat(all_fold_trades, ignore_index=True)
            combined_trades_log_path = os.path.join(self.config.get('base_dir', '.'), "walk_forward_combined_trades.csv")
            try:
                combined_trades_log.to_csv(combined_trades_log_path, index=False)
                print(f"WFO combined trades log saved to {combined_trades_log_path}")
            except Exception as e:
                print(f"Error saving WFO combined trades log: {e}")

        # Plot combined equity curve (more complex, requires careful stitching of equity from each fold)
        if all_fold_equity_curves and self.visualizer:
            try:
                # This is a simplified stitching. True portfolio equity requires compounding.
                # For now, just concatenate and plot.
                # A better way is to simulate portfolio equity based on trades.
                combined_equity_df = pd.concat(all_fold_equity_curves).drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

                if not combined_equity_df.empty:
                    # Re-base the equity curve to start from initial_balance for a continuous view (approximate)
                    initial_overall_balance = self.config.get('initial_balance', 10000)
                    # This is a visual re-basing, not a true compounded portfolio equity.
                    # For a simple plot:
                    # combined_equity_df['equity_plot'] = initial_overall_balance + (combined_equity_df['equity'] - combined_equity_df['equity'].iloc[0])

                    # For now, we'll plot the concatenated raw equity segments.
                    print("Plotting concatenated WFO equity segments (approximate view)...")
                    # Use the first fold's initial balance for the plot's reference initial balance
                    plot_initial_balance = self.walk_forward_results_summary[0]['initial_equity'] if self.walk_forward_results_summary else self.config.get('initial_balance', 10000)
                    plot_final_balance = combined_equity_df['equity'].iloc[-1] if not combined_equity_df.empty else plot_initial_balance
                    plot_return_pct = ((plot_final_balance - plot_initial_balance) / plot_initial_balance * 100) if plot_initial_balance != 0 else 0

                    self.visualizer.plot_equity_curve(
                        combined_equity_df,
                        initial_balance=plot_initial_balance,
                        final_balance=plot_final_balance,
                        total_return_pct=plot_return_pct,  # This is not the true compounded return
                        plot_type="WalkForward_Combined"
                    )
            except Exception as e:
                print(f"Error plotting combined WFO equity curve: {e}")
                traceback.print_exc()

        print("Walk-Forward Optimization finished.")
        return True

    def run_live_simulation(self):
        # (No changes from previous version for Phase 1, but ensure it uses the latest trained model)
        if not self.live_simulator:
            print("Error (Orchestrator): LiveSimulator not initialized.")
            return False
        if not self.model_predictor:
            print("Error (Orchestrator): ModelPredictor not initialized for live simulation.")
            return False

        print("\n--- Starting Live Simulation ---")
        use_ensemble_for_sim = self.config.get('use_ensemble_for_simulation', self.config.get('train_ensemble', False))
        # Load the model that was just trained (or specified in config)
        if not self.model_predictor.load_model_and_features(load_ensemble=use_ensemble_for_sim):
            print("Error (Orchestrator): Failed to load model for live simulation.")
            return False

        # Ensure scaling params are set in predictor
        if self.target_mean_for_prediction is not None and self.target_std_for_prediction is not None:
            self.model_predictor.set_scaling_params(self.target_mean_for_prediction, self.target_std_for_prediction)
        elif not self.model_predictor._ensure_scaling_info():
            print("Warning (Orchestrator): Could not ensure scaling info for simulation. Predictions may be affected.")

        sim_duration = self.config.get('simulation_duration_seconds', 300)
        self.live_simulator.start_live_simulation()

        try:
            start_sim_time = time.time()
            while self.live_simulator.simulation_running and (time.time() - start_sim_time) < sim_duration:
                print(f"Simulation running... (Time elapsed: {int(time.time() - start_sim_time)}s / {sim_duration}s)", end='\r')
                time.sleep(5)
            print("\nSimulation duration reached or stop signaled.")
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Stopping simulation...")
        finally:
            if self.live_simulator.simulation_running:
                self.live_simulator.stop_live_simulation()
            if self.visualizer:
                df_sim_equity = self.live_simulator.get_simulation_equity_data()
                if not df_sim_equity.empty:
                    initial_sim_bal = df_sim_equity['equity'].iloc[0] if not df_sim_equity.empty else self.config.get('initial_balance', 10000)
                    final_sim_bal = df_sim_equity['equity'].iloc[-1] if not df_sim_equity.empty else initial_sim_bal
                    sim_return_pct = ((final_sim_bal - initial_sim_bal) / initial_sim_bal * 100) if initial_sim_bal != 0 else 0
                    self.visualizer.plot_equity_curve(df_sim_equity, initial_sim_bal, final_sim_bal, sim_return_pct, "Simulation")
                else:
                    print("No simulation equity data to plot.")
        return True

    def visualize_results(self):
        # (No changes from previous version for Phase 1, but ensure it uses the latest trained model info)
        if not self.visualizer:
            print("Error (Orchestrator): Visualizer not initialized.")
            return

        print("\n--- Visualizing Results ---")
        model_to_plot_importance = None
        is_ensemble_clf_for_plot = False
        features_for_plot = self.trained_features_list  # Use features from last training

        if self.trained_ensemble_models and 'classifier' in self.trained_ensemble_models:
            model_to_plot_importance = self.trained_ensemble_models['classifier']
            is_ensemble_clf_for_plot = True
        elif self.trained_model_booster:
            model_to_plot_importance = self.trained_model_booster

        if model_to_plot_importance and features_for_plot:
            self.visualizer.plot_feature_importance(model_to_plot_importance, features_for_plot, is_ensemble_classifier=is_ensemble_clf_for_plot)
        else:
            print("Warning (Orchestrator): No trained model or features to plot importance for. Attempting to load last used model.")
            use_ensemble = self.config.get('use_ensemble_for_visualization', self.config.get('train_ensemble', False))
            if self.model_predictor and self.model_predictor.load_model_and_features(load_ensemble=use_ensemble):
                model_obj = self.model_predictor.model_object
                features = self.model_predictor.trained_features
                if use_ensemble and isinstance(model_obj, dict) and 'classifier' in model_obj:
                    self.visualizer.plot_feature_importance(model_obj['classifier'], features, is_ensemble_classifier=True)
                elif not use_ensemble and model_obj:
                    self.visualizer.plot_feature_importance(model_obj, features, is_ensemble_classifier=False)
                else:
                    print("Could not load a suitable model for feature importance visualization.")
            else:
                print("Failed to load any model for feature importance visualization.")

        data_for_plots = self.df_features if self.df_features is not None and not self.df_features.empty else None

        self.visualizer.plot_emd_decomposition(df_data=data_for_plots)
        self.visualizer.plot_features_vs_price(df_data=data_for_plots)

        print("Visualization generation attempted.")

    def run_full_workflow(self, run_wfo=False):  # Added run_wfo flag
        """
        Runs the full workflow: data prep, training, backtesting, and optional simulation.
        """
        if not self.exchange and not self.config.get('allow_no_exchange_init', False):
            print("FATAL (Orchestrator): Exchange not initialized and not allowed. Cannot run workflow.")
            return

        if run_wfo:  # --- Phase 1b: Execute Walk-Forward Optimization ---
            if not self.run_walk_forward_optimization():
                print("Walk-Forward Optimization failed. Aborting further steps in this workflow.")
                return
            # After WFO, you might want to train a final model on all data, or use the last fold's model.
            # For now, WFO is an alternative evaluation. If you want to proceed to live sim after WFO,
            # you'd need to decide which model to use (e.g., train one last time on all data).
            print("WFO complete. To proceed with live simulation, ensure a final model is trained/loaded appropriately.")
        else:  # Standard workflow
            if not self.prepare_data_for_training():
                return
            if not self.train_model():
                return
            # Run backtest on the full dataset using the model just trained
            self.run_backtest(df_backtest_data=self.df_features, load_latest_model=False)  # Use current model in predictor

        if self.config.get('run_simulation_flag', False) and not run_wfo:  # Only run sim if not in WFO mode (or handle model choice post-WFO)
            if self.live_simulator:
                self.run_live_simulation()
            else:
                print("Warning (Orchestrator): Live simulation flagged to run, but LiveSimulator component is not available.")

        if not run_wfo:  # Visualize results of standard workflow
            self.visualize_results()

        print("\n--- Full Workflow Finished ---")