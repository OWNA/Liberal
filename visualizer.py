"""
Module: visualizer
Description: Handles the generation and saving of various plots for trading bot analysis.
Author: project
Date: 2024-06-09
"""

import os
import pandas as pd
import numpy as np
import traceback
from typing import Any, Dict, List, Optional


class Visualizer:
    """
    Handles the generation and saving of various plots for trading bot analysis.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        has_matplotlib: bool = False,
        plt_module: Any = None,
        has_shap: bool = False,
        shap_module: Any = None,
        has_pyemd: bool = False,
        emd_class: Any = None,
        has_scipy_hilbert: bool = False,
        hilbert_func: Any = None
    ) -> None:
        """
        Initializes the Visualizer.

        Args:
            config: Configuration dictionary.
            has_matplotlib: Flag if matplotlib is available.
            plt_module: The imported matplotlib.pyplot module.
            has_shap: Flag if SHAP is available.
            shap_module: The imported SHAP module.
            has_pyemd: Flag if PyEMD is available.
            emd_class: The imported PyEMD.EMD class.
            has_scipy_hilbert: Flag if scipy.signal.hilbert is available.
            hilbert_func: The imported scipy.signal.hilbert function.
        """
        self.config = config
        self.symbol = config.get('symbol', 'SYMBOL')
        self.timeframe = config.get('timeframe', 'TIMEFRAME')
        self.base_dir = config.get('base_dir', './trading_bot_data')

        self.HAS_MATPLOTLIB = has_matplotlib
        self.plt = plt_module
        self.HAS_SHAP = has_shap
        self.shap = shap_module
        self.HAS_PYEMD = has_pyemd
        self.EMD = emd_class
        self.HAS_SCIPY_HILBERT = has_scipy_hilbert
        self.hilbert = hilbert_func

        safe_symbol = self.symbol.replace('/', '_').replace(':', '')
        plot_dir = os.path.join(self.base_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        self.imf_plot_path = os.path.join(
            plot_dir,
            f"emd_imf_plot_{safe_symbol}_{self.timeframe}.png"
        )
        self.feature_plot_path = os.path.join(
            plot_dir,
            f"feature_vs_price_plot_{safe_symbol}_{self.timeframe}.png"
        )
        self.shap_bar_plot_path = os.path.join(
            plot_dir,
            f"shap_bar_plot_{safe_symbol}_{self.timeframe}.png"
        )
        self.shap_dot_plot_path = os.path.join(
            plot_dir,
            f"shap_dot_plot_{safe_symbol}_{self.timeframe}.png"
        )
        self.lgbm_importance_plot_path = os.path.join(
            plot_dir,
            f"lgbm_importance_plot_{safe_symbol}_{self.timeframe}.png"
        )
        self.backtest_equity_plot_path = os.path.join(
            plot_dir,
            f"backtest_equity_curve_{safe_symbol}_{self.timeframe}.png"
        )
        self.simulation_equity_plot_path = os.path.join(
            plot_dir,
            f"simulation_equity_curve_{safe_symbol}_{self.timeframe}.png"
        )
        self.prepared_data_path = os.path.join(
            self.base_dir,
            f"prepared_data_{safe_symbol}_{self.timeframe}.csv"
        )

        if self.HAS_MATPLOTLIB and self.plt:
            pass  # Optionally set global style here
        print("Visualizer initialized.")

    def plot_equity_curve(
        self,
        equity_df: pd.DataFrame,
        initial_balance: float,
        final_balance: float,
        total_return_pct: float,
        plot_type: str = "Backtest"
    ) -> None:
        """
        Generates and displays/saves a plot of the equity curve.
        """
        if not (self.HAS_MATPLOTLIB and self.plt):
            print(
                "Warning (Visualizer): Matplotlib not available. "
                "Cannot plot equity curve."
            )
            return
        if (
            equity_df is None or equity_df.empty or
            'timestamp' not in equity_df.columns or
            'equity' not in equity_df.columns
        ):
            print(
                "Warning (Visualizer): Equity DataFrame is invalid or missing "
                "required columns for plotting."
            )
            return
        if len(equity_df) <= 1 and plot_type != "Simulation":
            if not (
                plot_type == "Simulation" and
                len(equity_df) == 1 and
                equity_df['equity'].iloc[0] == initial_balance
            ):
                print(
                    "Warning (Visualizer): Not enough data points to plot equity "
                    "curve."
                )
                return
        try:
            self.plt.figure(
                figsize=self.config.get('plot_figsize_equity', (14, 7))
            )
            equity_df_plot = equity_df.copy()
            equity_df_plot['timestamp'] = pd.to_datetime(equity_df_plot['timestamp'])
            self.plt.plot(
                equity_df_plot["timestamp"],
                equity_df_plot["equity"],
                label="Equity Curve",
                color='dodgerblue',
                lw=1.5
            )
            if len(equity_df_plot) > 1:
                self.plt.fill_between(
                    equity_df_plot["timestamp"],
                    initial_balance,
                    equity_df_plot["equity"],
                    where=equity_df_plot["equity"] >= initial_balance,
                    color='palegreen',
                    alpha=0.5,
                    interpolate=True
                )
                self.plt.fill_between(
                    equity_df_plot["timestamp"],
                    initial_balance,
                    equity_df_plot["equity"],
                    where=equity_df_plot["equity"] < initial_balance,
                    color='lightcoral',
                    alpha=0.5,
                    interpolate=True
                )
            self.plt.axhline(
                initial_balance,
                color='grey',
                linestyle='--',
                label=f'Initial Balance (${initial_balance:,.2f})'
            )
            title = (
                f"{plot_type} Equity: {self.symbol} {self.timeframe} | "
                f"Final: ${final_balance:,.2f} ({total_return_pct:.2f}%)"
            )
            self.plt.title(title, fontsize=14)
            self.plt.xlabel("Time", fontsize=12)
            self.plt.ylabel("Equity (USD)", fontsize=12)
            self.plt.grid(True, linestyle=':', alpha=0.7)
            self.plt.legend(fontsize=10)
            self.plt.xticks(rotation=45, ha='right')
            self.plt.tight_layout()
            save_path = (
                self.backtest_equity_plot_path
                if plot_type == "Backtest"
                else self.simulation_equity_plot_path
            )
            self.plt.savefig(save_path, dpi=150)
            print(f"{plot_type} equity curve saved to {save_path}")
            if self.config.get('show_plots', True):
                self.plt.show()
            self.plt.close()
        except Exception as e:
            print(f"Warning (Visualizer): Error plotting equity curve: {e}")
            traceback.print_exc()

    def plot_feature_importance(
        self,
        model_booster: Any,
        trained_features: List[str],
        use_shap_override: Optional[bool] = None,
        is_ensemble_classifier: bool = False
    ) -> None:
        """
        Visualizes feature importance for the trained model.
        """
        if not (self.HAS_MATPLOTLIB and self.plt):
            print(
                "Warning (Visualizer): Matplotlib not available. "
                "Cannot plot feature importance."
            )
            return
        if model_booster is None or not trained_features:
            print(
                "Warning (Visualizer): Model or trained features not provided "
                "for importance plot."
            )
            return
        use_shap = (
            use_shap_override
            if use_shap_override is not None
            else self.config.get('use_shap_for_importance', True)
        )
        if use_shap and not (self.HAS_SHAP and self.shap):
            print(
                "Warning (Visualizer): SHAP requested but not available. "
                "Falling back to LightGBM importance."
            )
            use_shap = False
        plot_title_prefix = (
            "Ensemble Classifier" if is_ensemble_classifier else "Standard Model"
        )
        if use_shap:
            print("Attempting SHAP feature importance plot...")
            try:
                if not os.path.exists(self.prepared_data_path):
                    print(
                        f"Error (Visualizer): Prepared data file not found at "
                        f"{self.prepared_data_path} for SHAP. Falling back."
                    )
                    self.plot_feature_importance(
                        model_booster,
                        trained_features,
                        use_shap_override=False,
                        is_ensemble_classifier=is_ensemble_classifier
                    )
                    return
                df_shap_data_full = pd.read_csv(
                    self.prepared_data_path,
                    usecols=trained_features + ['target']
                )
                df_shap_data = df_shap_data_full[trained_features].copy()
                df_shap_data.dropna(inplace=True)
                if df_shap_data.empty:
                    print(
                        "Error (Visualizer): No valid data for SHAP after loading "
                        "and NaN drop. Falling back."
                    )
                    self.plot_feature_importance(
                        model_booster,
                        trained_features,
                        use_shap_override=False,
                        is_ensemble_classifier=is_ensemble_classifier
                    )
                    return
                max_shap_samples = self.config.get('shap_max_samples', 2000)
                if len(df_shap_data) > max_shap_samples:
                    df_shap_data = df_shap_data.sample(
                        max_shap_samples,
                        random_state=self.config.get('random_state', 42)
                    )
                explainer = self.shap.Explainer(model_booster, df_shap_data)
                shap_values = explainer(df_shap_data)
                self.plt.figure()
                self.shap.summary_plot(
                    shap_values, df_shap_data, plot_type="bar", show=False
                )
                self.plt.title(f"SHAP Bar Importance - {plot_title_prefix}")
                self.plt.tight_layout()
                self.plt.savefig(self.shap_bar_plot_path, dpi=150)
                print(f"SHAP bar plot saved to {self.shap_bar_plot_path}")
                if self.config.get('show_plots', True):
                    self.plt.show()
                self.plt.close()
                self.plt.figure()
                self.shap.summary_plot(
                    shap_values, df_shap_data, show=False
                )
                self.plt.title(f"SHAP Dot Summary - {plot_title_prefix}")
                self.plt.tight_layout()
                self.plt.savefig(self.shap_dot_plot_path, dpi=150)
                print(f"SHAP dot plot saved to {self.shap_dot_plot_path}")
                if self.config.get('show_plots', True):
                    self.plt.show()
                self.plt.close()
                return
            except Exception as e:
                print(
                    f"Warning (Visualizer): SHAP plot failed: {e}. "
                    "Falling back to LightGBM importance."
                )
                traceback.print_exc()
        print("Using LightGBM feature importance plot.")
        try:
            self.plt.figure(
                figsize=self.config.get('plot_figsize_lgbm', (12, 10))
            )
            # lgb is not imported here; user must ensure it's available in context
            if hasattr(model_booster, 'feature_importances_'):
                importances = model_booster.feature_importances_
                feature_names = np.array(trained_features)
                sorted_indices = np.argsort(importances)[::-1]
                num_to_plot = min(len(trained_features), 30)
                self.plt.barh(
                    range(num_to_plot),
                    importances[sorted_indices][:num_to_plot][::-1],
                    align="center"
                )
                self.plt.yticks(
                    range(num_to_plot),
                    feature_names[sorted_indices][:num_to_plot][::-1]
                )
                self.plt.xlabel("Feature Importance (Gain/Default)")
            else:
                print(
                    "Error (Visualizer): Model type not supported for "
                    "LightGBM-style importance plot."
                )
                return
            self.plt.title(
                f"LightGBM Importance - {plot_title_prefix}", fontsize=14
            )
            self.plt.tight_layout()
            self.plt.savefig(self.lgbm_importance_plot_path, dpi=150)
            print(f"LGBM importance plot saved to {self.lgbm_importance_plot_path}")
            if self.config.get('show_plots', True):
                self.plt.show()
            self.plt.close()
        except Exception as e:
            print(f"Warning (Visualizer): LightGBM plot error: {e}")
            traceback.print_exc()

    def plot_emd_decomposition(
        self,
        df_data: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Plots EMD of the close price.
        """
        if not (
            self.HAS_PYEMD and self.HAS_SCIPY_HILBERT and self.HAS_MATPLOTLIB and
            self.plt and self.EMD and self.hilbert
        ):
            print(
                "Warning (Visualizer): Missing libraries for EMD plot "
                "(PyEMD, Scipy, Matplotlib)."
            )
            return
        plot_df = df_data
        if plot_df is None or 'close' not in plot_df.columns:
            if os.path.exists(self.prepared_data_path):
                print(
                    f"Loading data from {self.prepared_data_path} for EMD plot."
                )
                plot_df = pd.read_csv(
                    self.prepared_data_path,
                    parse_dates=['timestamp'],
                    usecols=['timestamp', 'close']
                )
            else:
                print(
                    f"Error (Visualizer): Prepared data file not found at "
                    f"{self.prepared_data_path} for EMD plot."
                )
                return
        if (
            plot_df is None or plot_df.empty or
            'close' not in plot_df.columns or
            'timestamp' not in plot_df.columns
        ):
            print("Error (Visualizer): Invalid DataFrame for EMD plot.")
            return
        signal = plot_df['close'].values.astype(float)
        time_vec = pd.to_datetime(plot_df['timestamp']).values
        feature_window = self.config.get('feature_window', 20)
        if len(signal) < feature_window + 20:
            print("Warning (Visualizer): Not enough data for EMD plot.")
            return
        try:
            emd_instance = self.EMD(
                DTYPE=np.float64,
                noise_width=self.config.get('hht_emd_noise_width', 0.05)
            )
            imfs_and_residue = emd_instance(signal)
            if imfs_and_residue is None or imfs_and_residue.shape[0] < 1:
                print("Warning (Visualizer): EMD failed to produce IMFs.")
                return
            num_components = imfs_and_residue.shape[0]
            num_imfs = (
                num_components - 1
                if num_components > 1 and np.any(imfs_and_residue[-1])
                else num_components
            )
            fig, axes = self.plt.subplots(
                num_components + 1,
                1,
                figsize=self.config.get(
                    'plot_figsize_emd', (14, 2.5 * (num_components + 1))
                ),
                sharex=True
            )
            axes[0].plot(time_vec, signal, color='grey', lw=1.5)
            axes[0].set_ylabel("Original Signal")
            axes[0].grid(True, linestyle=':', alpha=0.6)
            for i in range(num_components):
                label = f"IMF {i+1}" if i < num_imfs else "Residue"
                axes[i+1].plot(time_vec, imfs_and_residue[i, :], lw=1)
                axes[i+1].set_ylabel(label)
                axes[i+1].grid(True, linestyle=':', alpha=0.6)
            axes[-1].set_xlabel("Time")
            fig.align_ylabels(axes)
            fig.suptitle(
                f"EMD Decomposition - {self.symbol} {self.timeframe}", fontsize=14
            )
            self.plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            self.plt.savefig(self.imf_plot_path, dpi=150)
            print(f"EMD decomposition plot saved to {self.imf_plot_path}")
            if self.config.get('show_plots', True):
                self.plt.show()
            self.plt.close()
        except Exception as e:
            print(f"Warning (Visualizer): Error plotting EMD: {e}")
            traceback.print_exc()

    def plot_features_vs_price(
        self,
        df_data: Optional[pd.DataFrame] = None,
        features_to_plot: Optional[List[str]] = None
    ) -> None:
        """
        Plots selected features against the close price.
        """
        if not (self.HAS_MATPLOTLIB and self.plt):
            print(
                "Warning (Visualizer): Matplotlib not available. "
                "Cannot plot features vs price."
            )
            return
        _features_to_plot = features_to_plot
        if _features_to_plot is None:
            _features_to_plot = []
            if self.HAS_PYEMD and self.HAS_SCIPY_HILBERT:
                _features_to_plot.extend([
                    f'{fb}0' for fb in self.config.get(
                        'hht_features_imf_bases', ['hht_freq_imf', 'hht_amp_imf']
                    )[:1]
                ])
            if getattr(self, 'HAS_PANDAS_TA', False):
                _features_to_plot.extend([
                    f for f in ['rsi', 'atr', 'vwap', 'supertrend', 'kama']
                    if f in self.config.get('ta_features', [])
                ])
            if self.config.get('use_l2_features', False):
                _features_to_plot.extend(
                    self.config.get('l2_features', [])[:2]
                )
            if not _features_to_plot:
                _features_to_plot = [
                    f for f in self.config.get('ohlcv_base_features', [])
                    if f.startswith('z_')
                ]
            _features_to_plot = sorted(list(set(_features_to_plot)))
            if not _features_to_plot:
                print(
                    "Warning (Visualizer): No default features available to plot "
                    "for features_vs_price."
                )
                return
        plot_df = df_data
        required_cols_for_plot = list(set(['timestamp', 'close'] + _features_to_plot))
        if (
            plot_df is None or
            not all(col in plot_df.columns for col in required_cols_for_plot)
        ):
            if os.path.exists(self.prepared_data_path):
                print(
                    f"Loading data from {self.prepared_data_path} for features vs price plot."
                )
                try:
                    plot_df = pd.read_csv(
                        self.prepared_data_path,
                        parse_dates=['timestamp'],
                        usecols=lambda c: c in required_cols_for_plot
                    )
                except ValueError as ve:
                    print(
                        f"Could not load all required columns from prepared_data.csv: {ve}. "
                        "Trying with available columns."
                    )
                    plot_df = pd.read_csv(
                        self.prepared_data_path,
                        parse_dates=['timestamp']
                    )
                    _features_to_plot = [
                        f for f in _features_to_plot if f in plot_df.columns
                    ]
                    required_cols_for_plot = list(
                        set(['timestamp', 'close'] + _features_to_plot)
                    )
            else:
                print(
                    f"Error (Visualizer): Prepared data file not found at "
                    f"{self.prepared_data_path}."
                )
                return
        if plot_df is None or plot_df.empty:
            print("Error (Visualizer): DataFrame empty for features vs price plot.")
            return
        actual_features_present = [
            f for f in _features_to_plot if f in plot_df.columns
        ]
        if not actual_features_present:
            print(
                f"Error (Visualizer): None of the requested features "
                f"({_features_to_plot}) found in the data."
            )
            return
        plot_df_cleaned = plot_df[
            ['timestamp', 'close'] + actual_features_present
        ].copy()
        plot_df_cleaned.dropna(
            subset=['timestamp', 'close'] + actual_features_present,
            inplace=True
        )
        if plot_df_cleaned.empty:
            print(
                "Error (Visualizer): DataFrame empty after dropping NaNs from "
                "required plot columns."
            )
            return
        plot_df_cleaned['timestamp'] = pd.to_datetime(plot_df_cleaned['timestamp'])
        try:
            num_plots = 1 + len(actual_features_present)
            fig, axes = self.plt.subplots(
                num_plots,
                1,
                figsize=self.config.get('plot_figsize_features', (14, 3 * num_plots)),
                sharex=True
            )
            if num_plots == 1:
                axes = [axes]
            axes[0].plot(
                plot_df_cleaned['timestamp'],
                plot_df_cleaned['close'],
                label='Close Price',
                color='blue',
                lw=1.2
            )
            axes[0].set_ylabel('Price (USD)')
            axes[0].set_title(
                f'Price & Feature Interaction - {self.symbol} {self.timeframe}',
                fontsize=14
            )
            axes[0].legend(loc='upper left')
            axes[0].grid(True, linestyle=':', alpha=0.6)
            colors = self.plt.cm.viridis(
                np.linspace(0, 0.9, len(actual_features_present))
            )
            for i, feat_name in enumerate(actual_features_present):
                ax_idx = i + 1
                axes[ax_idx].plot(
                    plot_df_cleaned['timestamp'],
                    plot_df_cleaned[feat_name],
                    label=feat_name,
                    color=colors[i],
                    lw=1
                )
                axes[ax_idx].set_ylabel(feat_name.replace('_', ' ').title())
                axes[ax_idx].legend(loc='upper left')
                axes[ax_idx].grid(True, linestyle=':', alpha=0.6)
                if any(
                    k in feat_name for k in [
                        'z_', 'freq', 'rsi', 'imb', 'osc', 'norm', 'signal'
                    ]
                ):
                    axes[ax_idx].axhline(0, color='grey', linestyle=':', lw=0.8)
            axes[-1].set_xlabel("Time")
            axes[-1].tick_params(axis='x', rotation=45)
            fig.align_ylabels(axes)
            self.plt.tight_layout()
            self.plt.savefig(self.feature_plot_path, dpi=150)
            print(f"Features vs Price plot saved to {self.feature_plot_path}")
            if self.config.get('show_plots', True):
                self.plt.show()
            self.plt.close()
        except Exception as e:
            print(f"Warning (Visualizer): Error plotting features vs price: {e}")
            traceback.print_exc()