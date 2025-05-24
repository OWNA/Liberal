# label_generator.py
# Reformatted from notebook export to standard Python file

import pandas as pd
import numpy as np
import traceback

class LabelGenerator:
    """
    Generates target labels for model training from OHLCV data using various
    methods.
    """
    def __init__(self, config):
        """
        Initializes the LabelGenerator.
        """
        self.config = config
        self.labeling_method = config.get(
            'labeling_method', 'volatility_normalized_return'
        )

        # Parameters for 'volatility_normalized_return'
        self.vol_norm_volatility_window = config.get('label_volatility_window', 20)
        self.vol_norm_clip_lower_quantile = config.get(
            'label_clip_quantiles', (0.01, 0.99)
        )[0]
        self.vol_norm_clip_upper_quantile = config.get(
            'label_clip_quantiles', (0.01, 0.99)
        )[1]
        self.vol_norm_label_shift = config.get('label_shift', -1)

        # Parameters for 'triple_barrier'
        self.tb_profit_target_atr_mult = config.get(
            'triple_barrier_profit_target_atr_mult', 2.0
        )
        self.tb_stop_loss_atr_mult = config.get(
            'triple_barrier_stop_loss_atr_mult', 1.5
        )
        self.tb_time_horizon_bars = config.get(
            'triple_barrier_time_horizon_bars', 10
        )
        # ATR for triple barrier typically needs to be pre-calculated and present in df_features
        self.tb_atr_column = config.get(
            'triple_barrier_atr_column', 'atr'
        )  # Column name for ATR

        # Store these for potential use by ModelPredictor if it needs to re-calculate them
        # For vol_norm_return, these are mean/std of the scaled target.
        # For triple_barrier, these might be less relevant or represent class distribution.
        self.target_mean_calculated = None
        self.target_std_calculated = None

        print(
            f"LabelGenerator initialized (Phase 1 Update). Using method: "
            f"{self.labeling_method}"
        )

    def _generate_volatility_normalized_return_labels(self, df_features):
        """
        Generates volatility-normalized, clipped future returns.
        """
        if 'close' not in df_features.columns:
            print(
                "Error (LabelGenerator - VolNorm): 'close' column not in "
                "DataFrame."
            )
            return df_features.copy(), None, None

        df = df_features.copy()
        returns = df['close'].pct_change()
        min_periods_vol = max(1, self.vol_norm_volatility_window // 2)
        volatility = returns.rolling(
            window=self.vol_norm_volatility_window, min_periods=min_periods_vol
        ).std()
        df['target'] = returns.shift(self.vol_norm_label_shift) / (volatility + 1e-9)

        target_mean_raw = None
        target_std_raw = None

        if df['target'].notna().sum() > self.vol_norm_volatility_window:
            target_mean_raw = df['target'].mean()
            target_std_raw = df['target'].std()
            if (
                target_std_raw == 0 or not pd.notna(target_std_raw)
                or target_std_raw < 1e-9
            ):
                target_std_raw = 1e-9

            if pd.notna(self.vol_norm_clip_lower_quantile) and pd.notna(
                self.vol_norm_clip_upper_quantile
            ):
                lower_bound = df['target'].quantile(self.vol_norm_clip_lower_quantile)
                upper_bound = df['target'].quantile(self.vol_norm_clip_upper_quantile)
                if (
                    pd.notna(lower_bound) and pd.notna(upper_bound)
                    and lower_bound < upper_bound
                ):
                    df['target'] = df['target'].clip(lower_bound, upper_bound)
                else:
                    print(
                        "Warning (LabelGenerator - VolNorm): Could not clip "
                        "target due to invalid quantile bounds."
                    )
        else:
            print(
                "Warning (LabelGenerator - VolNorm): Not enough non-NaN target "
                "values to calculate mean/std or clip robustly."
            )
            target_mean_raw = 0.0
            target_std_raw = 1.0

        self.target_mean_calculated = target_mean_raw
        self.target_std_calculated = target_std_raw
        df.dropna(subset=['target'], inplace=True)
        return df, self.target_mean_calculated, self.target_std_calculated

    def _generate_triple_barrier_labels(self, df_features):
        """
        Generates labels using the Triple-Barrier Method.
        Target: 1 (profit target hit), -1 (stop-loss hit), 0 (time barrier or no
        barrier hit).
        Requires 'close', 'high', 'low', and an ATR column (e.g., 'atr').
        """
        required_cols = ['close', 'high', 'low', self.tb_atr_column]
        if not all(col in df_features.columns for col in required_cols):
            print(
                f"Error (LabelGenerator - TripleBarrier): Missing one or more "
                f"required columns: {required_cols}"
            )
            return df_features.copy(), None, None

        if df_features[self.tb_atr_column].isnull().any():
            print(
                f"Warning (LabelGenerator - TripleBarrier): ATR column "
                f"'{self.tb_atr_column}' contains NaNs. Rows with NaN ATR will "
                f"not have labels."
            )
            # Consider df_features.dropna(subset=[self.tb_atr_column], inplace=True) or fill
            # For now, proceed and NaNs in ATR will lead to NaN labels for those rows.

        df = df_features.copy()
        df['target'] = 0  # Default to 0 (time barrier / no hit)
        df['event_time'] = pd.NaT  # Time when a barrier was hit

        for i in range(len(df) - self.tb_time_horizon_bars):
            entry_price = df['close'].iloc[i]
            atr_at_entry = df[self.tb_atr_column].iloc[i]

            if pd.isna(entry_price) or pd.isna(atr_at_entry) or atr_at_entry <= 1e-9:
                df.loc[df.index[i], 'target'] = np.nan  # Cannot determine barriers
                continue

            upper_barrier = entry_price + (
                atr_at_entry * self.tb_profit_target_atr_mult
            )
            lower_barrier = entry_price - (
                atr_at_entry * self.tb_stop_loss_atr_mult
            )

            # Look ahead for barrier hits
            for k in range(1, self.tb_time_horizon_bars + 1):
                future_idx = i + k
                if future_idx >= len(df):
                    break  # Out of bounds

                future_high = df['high'].iloc[future_idx]
                future_low = df['low'].iloc[future_idx]
                future_timestamp = df['timestamp'].iloc[future_idx]

                # Check profit target hit (for a long-biased scenario; adapt if shorting)
                if future_high >= upper_barrier:
                    df.loc[df.index[i], 'target'] = 1
                    df.loc[df.index[i], 'event_time'] = future_timestamp
                    break
                # Check stop-loss hit
                if future_low <= lower_barrier:
                    df.loc[df.index[i], 'target'] = -1
                    df.loc[df.index[i], 'event_time'] = future_timestamp
                    break
            # If loop completes without a hit, target remains 0 (time barrier)
            # If no event time was set, it means time barrier was hit implicitly at tb_time_horizon_bars
            if pd.isna(df.loc[df.index[i], 'event_time']) and not pd.isna(
                df.loc[df.index[i], 'target']
            ):
                if (i + self.tb_time_horizon_bars) < len(df):
                    df.loc[df.index[i], 'event_time'] = df['timestamp'].iloc[
                        i + self.tb_time_horizon_bars
                    ]
                # else: event_time remains NaT, target is 0

        # For triple barrier, mean/std of target is less directly used for scaling predictions
        # but we can calculate class distribution if needed.
        self.target_mean_calculated = None
        self.target_std_calculated = None

        # Drop rows where target could not be calculated (e.g., due to NaN ATR or end of series)
        df.dropna(subset=['target'], inplace=True)
        df['target'] = df['target'].astype(int)  # Ensure target is integer
        return df, self.target_mean_calculated, self.target_std_calculated

    def generate_labels(self, df_features):
        """
        Generates target labels based on the method specified in the config.
        """
        if df_features is None or df_features.empty:
            print("Error (LabelGenerator): Input DataFrame is empty.")
            return df_features, None, None

        print(f"Generating labels using method: {self.labeling_method}")

        if self.labeling_method == 'volatility_normalized_return':
            df_labeled, mean_val, std_val = self._generate_volatility_normalized_return_labels(df_features)
        elif self.labeling_method == 'triple_barrier':
            df_labeled, mean_val, std_val = self._generate_triple_barrier_labels(df_features)
        else:
            print(
                f"Error (LabelGenerator): Unknown labeling_method "
                f"'{self.labeling_method}'. Defaulting to 'volatility_normalized_return'."
            )
            df_labeled, mean_val, std_val = self._generate_volatility_normalized_return_labels(df_features)

        if df_labeled.empty:
            print(
                "Warning (LabelGenerator): DataFrame is empty after generating "
                "labels and dropping NaNs."
            )
        else:
            if self.labeling_method == 'volatility_normalized_return':
                print(
                    f"Labels generated. Target mean (raw scaled): {mean_val}, "
                    f"Target std (raw scaled): {std_val}"
                )
            elif self.labeling_method == 'triple_barrier':
                print(
                    "Triple-barrier labels generated. Class distribution:\n"
                    f"{df_labeled['target'].value_counts(normalize=True, dropna=False)}"
                )

        return df_labeled, mean_val, std_val