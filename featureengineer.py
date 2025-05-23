# featureengineer.py
# Fixed version with proper TA feature handling and NaN management

import os
import pandas as pd
import numpy as np
import json
import traceback

class FeatureEngineer:
    """
    Calculates various features from OHLCV and L2 order book data.
    Fixed version that properly handles multi-column TA outputs and NaN values.
    """
    def __init__(self, config, has_pandas_ta=False, has_pyemd=False, has_scipy_hilbert=False,
                 ta_module=None, emd_class=None, hilbert_func=None):
        """
        Initializes the FeatureEngineer.
        """
        self.config = config
        self.feature_window = config.get('feature_window', 24)

        self.HAS_PANDAS_TA = has_pandas_ta
        self.HAS_PYEMD = has_pyemd
        self.HAS_SCIPY_HILBERT = has_scipy_hilbert
        self.ta = ta_module
        self.EMD = emd_class
        self.hilbert = hilbert_func

        # --- Phase 1a: Read detailed TA parameters and L2 feature levels from config ---
        self.ta_indicator_params = config.get('ta_indicator_params', {})
        self.l2_depth_imbalance_levels = config.get('l2_depth_imbalance_levels', [5, 10, 20])

        self.ohlcv_base_features = config.get('ohlcv_base_features', ["z_close", "z_volume", "z_spread"])
        self.ta_features_config = config.get('ta_features', ['kama', 'supertrend', 'vwap', 'atr', 'rsi'])
        self.hht_features_imf_bases = config.get('hht_features_imf_bases', ['hht_freq_imf', 'hht_amp_imf'])
        self.hht_imf_count = config.get('hht_imf_count', 3)

        # Update L2 feature list generation if specific imbalance levels are used
        base_l2_features = [f for f in config.get('l2_features', []) if not f.startswith('depth_imb_')]
        dynamic_l2_imb_features = [f'depth_imb_{level}' for level in self.l2_depth_imbalance_levels]
        self.l2_features_list_config = sorted(list(set(base_l2_features + dynamic_l2_imb_features)))

        self.use_l2_features = config.get('use_l2_features', False)

        self.all_defined_feature_columns = list(self.ohlcv_base_features)
        if self.HAS_PANDAS_TA and self.ta_features_config:
            self.all_defined_feature_columns.extend(self.ta_features_config)
        if self.HAS_PYEMD and self.HAS_SCIPY_HILBERT and self.hht_features_imf_bases:
            for i in range(self.hht_imf_count):
                for base_feat in self.hht_features_imf_bases:
                    self.all_defined_feature_columns.append(f'{base_feat}{i}')
        if self.use_l2_features and self.l2_features_list_config:
             self.all_defined_feature_columns.extend(self.l2_features_list_config)
        self.all_defined_feature_columns = sorted(list(set(self.all_defined_feature_columns)))

        self.base_dir = config.get('base_dir', './trading_bot_data')
        safe_symbol = config.get('symbol', 'SYMBOL').replace('/', '_').replace(':', '')
        timeframe = config.get('timeframe', 'TIMEFRAME')
        self.prepared_data_path = os.path.join(self.base_dir, f"prepared_data_{safe_symbol}_{timeframe}.csv")

        print("FeatureEngineer initialized (Phase 1 Update).")

    def _calculate_zscore_features(self, df_input):
        """Calculate Z-score features with improved NaN handling."""
        df = df_input.copy()
        window = self.feature_window
        min_periods = max(1, min(window // 2, len(df) // 2))  # Adaptive min_periods
        
        for col in ["close", "volume"]:
            if col not in df.columns:
                print(f"Warning (FeatureEngineer): Column {col} not found for Z-score calculation.")
                df[f'z_{col}'] = np.nan
                continue
                
            # Calculate rolling statistics
            mean = df[col].rolling(window=window, min_periods=min_periods).mean()
            std = df[col].rolling(window=window, min_periods=min_periods).std()
            
            # Handle zero/NaN standard deviation
            std = std.fillna(1e-9).replace(0, 1e-9)
            
            # Calculate z-score
            df[f'z_{col}'] = (df[col] - mean) / std
            
            # Fill any remaining NaNs with forward fill, then backward fill, then 0
            df[f'z_{col}'] = df[f'z_{col}'].fillna(method='ffill').fillna(method='bfill').fillna(0.0)
            
        return df

    def _calculate_spread_features(self, df_input):
        """Calculate spread features with improved NaN handling."""
        df = df_input.copy()
        window = self.feature_window
        min_periods = max(1, min(window // 2, len(df) // 2))  # Adaptive min_periods
        
        if 'high' not in df.columns or 'low' not in df.columns:
            print(f"Warning (FeatureEngineer): Columns high/low not found for spread calculation.")
            df['spread'] = np.nan
            df['z_spread'] = np.nan
            return df

        # Calculate spread
        df['spread'] = df['high'] - df['low']
        
        # Handle cases where spread might be zero or very small
        df['spread'] = df['spread'].clip(lower=1e-9)  # Minimum spread to avoid division issues
        
        # Calculate rolling statistics
        mean = df['spread'].rolling(window=window, min_periods=min_periods).mean()
        std = df['spread'].rolling(window=window, min_periods=min_periods).std()
        
        # Handle zero/NaN standard deviation
        std = std.fillna(1e-9).replace(0, 1e-9)
        
        # Calculate z-score
        df['z_spread'] = (df['spread'] - mean) / std
        
        # Fill any remaining NaNs
        df['z_spread'] = df['z_spread'].fillna(method='ffill').fillna(method='bfill').fillna(0.0)
        
        return df

    def _calculate_advanced_ta_features(self, df_input):
        """Fixed TA features calculation with proper handling of multi-column outputs."""
        df = df_input.copy()
        if not self.HAS_PANDAS_TA or not self.ta:
            for feat_name in self.ta_features_config: 
                df[feat_name] = np.nan
            return df

        # Ensure we have a datetime index for TA calculations that need it
        df_ta = df.copy()
        if not isinstance(df_ta.index, pd.DatetimeIndex):
            # Try to use timestamp column if available
            if 'timestamp' in df_ta.columns:
                try:
                    df_ta.index = pd.to_datetime(df_ta['timestamp'])
                except:
                    # Create a simple datetime index
                    df_ta.index = pd.date_range(start='2024-01-01', periods=len(df_ta), freq='1min')
            else:
                # Create a simple datetime index for TA calculations
                df_ta.index = pd.date_range(start='2024-01-01', periods=len(df_ta), freq='1min')

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_ta.columns: 
                df_ta[col] = pd.to_numeric(df_ta[col], errors='coerce')
            else:
                print(f"Warning (FeatureEngineer): Core column {col} missing for TA features.")
                for feat_name in self.ta_features_config: 
                    df[feat_name] = np.nan
                return df

        # --- Phase 1a: Use detailed TA parameters from config ---
        for ta_name in self.ta_features_config:
            params = self.ta_indicator_params.get(ta_name, {}) # Get specific params or empty dict
            try:
                if ta_name == 'kama':
                    length = params.get('length', self.config.get('ta_kama_length', 20))
                    df['kama'] = self.ta.kama(df_ta['close'], length=length, **{k:v for k,v in params.items() if k!='length'})
                    
                elif ta_name == 'supertrend':
                    length = params.get('length', self.config.get('ta_supertrend_length', 10))
                    multiplier = params.get('multiplier', self.config.get('ta_supertrend_multiplier', 3))
                    atr_length = params.get('atr_period', params.get('atr_length', self.config.get('risk_management', {}).get('volatility_lookback', 14)))
                    
                    st = self.ta.supertrend(df_ta['high'], df_ta['low'], df_ta['close'], 
                                          length=atr_length, multiplier=multiplier)
                    if st is not None and not st.empty:
                        # Find the supertrend column - pandas_ta uses different naming conventions
                        st_col = None
                        for col_name in st.columns:
                            if 'SUPERT_' in col_name and 'SUPERTd_' not in col_name:
                                st_col = col_name
                                break
                        df['supertrend'] = st[st_col] if st_col else np.nan
                    else: 
                        df['supertrend'] = np.nan
                        
                elif ta_name == 'vwap':
                    try:
                        # VWAP needs proper datetime index and volume
                        if 'volume' in df_ta.columns:
                            # Ensure the data is properly sorted by index
                            df_vwap = df_ta.sort_index()
                            vwap_result = self.ta.vwap(df_vwap['high'], df_vwap['low'], 
                                                     df_vwap['close'], df_vwap['volume'], **params)
                            df['vwap'] = vwap_result
                        else:
                            print(f"Warning (FeatureEngineer): Volume data not available for VWAP, using price average.")
                            df['vwap'] = (df_ta['high'] + df_ta['low'] + df_ta['close']) / 3
                    except Exception as vwap_error:
                        print(f"Warning (FeatureEngineer): Error calculating VWAP: {vwap_error}")
                        # Use simple price average as fallback
                        df['vwap'] = (df_ta['high'] + df_ta['low'] + df_ta['close']) / 3
                        
                elif ta_name == 'atr':
                    length = params.get('length', self.config.get('risk_management', {}).get('volatility_lookback', 14))
                    df['atr'] = self.ta.atr(df_ta['high'], df_ta['low'], df_ta['close'], 
                                          length=length, **{k:v for k,v in params.items() if k!='length'})
                    
                elif ta_name == 'rsi':
                    length = params.get('length', self.config.get('ta_rsi_length', 14))
                    scalar = params.get('scalar', 100)
                    rsi_values = self.ta.rsi(df_ta['close'], length=length, 
                                           **{k:v for k,v in params.items() if k not in ['length', 'scalar']})
                    df['rsi'] = rsi_values / scalar if scalar != 100 else rsi_values
                    
                elif ta_name == 'macd':
                    # MACD returns a DataFrame with multiple columns - THIS IS THE KEY FIX
                    fast = params.get('fast', 12)
                    slow = params.get('slow', 26)
                    signal = params.get('signal', 9)
                    
                    macd_result = self.ta.macd(df_ta['close'], fast=fast, slow=slow, signal=signal)
                    
                    if isinstance(macd_result, pd.DataFrame) and not macd_result.empty:
                        # Extract individual MACD components
                        for col_name in macd_result.columns:
                            if f'MACD_{fast}_{slow}_{signal}' in col_name:
                                df['macd_line'] = macd_result[col_name]
                            elif f'MACDh_{fast}_{slow}_{signal}' in col_name:
                                df['macd_histogram'] = macd_result[col_name]
                            elif f'MACDs_{fast}_{slow}_{signal}' in col_name:
                                df['macd_signal'] = macd_result[col_name]
                        
                        # For backward compatibility, use the main MACD line as 'macd'
                        if 'macd_line' in df.columns:
                            df['macd'] = df['macd_line']
                        else:
                            df['macd'] = np.nan
                    else:
                        df['macd'] = np.nan
                        
                elif ta_name == 'bbands':
                    # Bollinger Bands returns a DataFrame with multiple columns - THIS IS THE KEY FIX
                    length = params.get('length', 20)
                    std = params.get('std', 2.0)
                    
                    bbands_result = self.ta.bbands(df_ta['close'], length=length, std=std)
                    
                    if isinstance(bbands_result, pd.DataFrame) and not bbands_result.empty:
                        # Extract individual Bollinger Band components
                        for col_name in bbands_result.columns:
                            if f'BBL_{length}_{std}' in col_name:
                                df['bb_lower'] = bbands_result[col_name]
                            elif f'BBM_{length}_{std}' in col_name:
                                df['bb_middle'] = bbands_result[col_name]
                            elif f'BBU_{length}_{std}' in col_name:
                                df['bb_upper'] = bbands_result[col_name]
                            elif f'BBB_{length}_{std}' in col_name:
                                df['bb_bandwidth'] = bbands_result[col_name]
                            elif f'BBP_{length}_{std}' in col_name:
                                df['bb_percent'] = bbands_result[col_name]
                        
                        # For backward compatibility, use the middle band as 'bbands'
                        if 'bb_middle' in df.columns:
                            df['bbands'] = df['bb_middle']
                        else:
                            df['bbands'] = np.nan
                    else:
                        df['bbands'] = np.nan
                        
                else:
                    # Generic attempt for other indicators
                    if hasattr(self.ta, ta_name):
                        indicator_func = getattr(self.ta, ta_name)
                        if 'close' in df_ta.columns:
                            result = indicator_func(df_ta['close'], **params)
                            # If result is a DataFrame, take the first column
                            if isinstance(result, pd.DataFrame):
                                df[ta_name] = result.iloc[:, 0] if not result.empty else np.nan
                            else:
                                df[ta_name] = result
                        else:
                            df[ta_name] = np.nan
                    else:
                        df[ta_name] = np.nan
                        print(f"Warning (FeatureEngineer): TA indicator '{ta_name}' not specifically handled or found in pandas_ta.")
                        
            except Exception as e:
                df[ta_name] = np.nan
                print(f"Warning (FeatureEngineer): Error calculating TA feature '{ta_name}': {e}")
                
        return df

    def _calculate_hht_features(self, df_input):
        """Calculate HHT features (unchanged from original)."""
        df = df_input.copy()
        hht_feature_names = [f'{base}{i}' for i in range(self.hht_imf_count) for base in self.hht_features_imf_bases]
        if not (self.HAS_PYEMD and self.HAS_SCIPY_HILBERT and self.EMD and self.hilbert):
            for col in hht_feature_names: df[col] = np.nan
            return df

        if 'close' not in df.columns:
            print("Warning (FeatureEngineer): Column 'close' not found for HHT calculation.")
            for col in hht_feature_names: df[col] = np.nan
            return df

        signal = df["close"].values.astype(float)
        for col in hht_feature_names: df[col] = np.nan

        if len(signal) < self.feature_window + 20 :
            print(f"Warning (FeatureEngineer): Signal length {len(signal)} too short for HHT.")
            return df
        try:
            emd_instance = self.EMD(noise_width=self.config.get('hht_emd_noise_width', 0.05), DTYPE=np.float64)
            imfs = emd_instance(signal)
            if imfs is None or imfs.shape[0] == 0: return df

            for i in range(min(self.hht_imf_count, imfs.shape[0])):
                analytic_signal = self.hilbert(imfs[i])
                instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                dt = 1
                instantaneous_frequency = np.gradient(instantaneous_phase) / (2.0 * np.pi * dt)
                instantaneous_amplitude = np.abs(analytic_signal)

                if len(self.hht_features_imf_bases) > 0 and f'{self.hht_features_imf_bases[0]}{i}' in hht_feature_names and len(instantaneous_frequency) == len(df):
                    df[f'{self.hht_features_imf_bases[0]}{i}'] = instantaneous_frequency
                if len(self.hht_features_imf_bases) > 1 and f'{self.hht_features_imf_bases[1]}{i}' in hht_feature_names and len(instantaneous_amplitude) == len(df):
                    df[f'{self.hht_features_imf_bases[1]}{i}'] = instantaneous_amplitude
        except Exception as e:
            print(f"Warning (FeatureEngineer) HHT calculation error: {e}")
            for col in hht_feature_names: df[col] = np.nan
        return df

    def calculate_l2_features_from_snapshot(self, bids_snapshot, asks_snapshot):
        """Calculate L2 features from snapshot (unchanged from original)."""
        features = {col: np.nan for col in self.l2_features_list_config}
        try:
            if not bids_snapshot or not asks_snapshot: return features

            bids_np = np.array([[float(p), float(v)] for p,v in bids_snapshot if p is not None and v is not None and float(v)>0])
            asks_np = np.array([[float(p), float(v)] for p,v in asks_snapshot if p is not None and v is not None and float(v)>0])

            if bids_np.ndim != 2 or bids_np.shape[1] != 2 or bids_np.shape[0] == 0: return features
            if asks_np.ndim != 2 or asks_np.shape[1] != 2 or asks_np.shape[0] == 0: return features

            for depth in self.l2_depth_imbalance_levels:
                feat_name = f'depth_imb_{depth}'
                if feat_name in self.l2_features_list_config:
                    cb = bids_np[:min(depth, len(bids_np))]; ca = asks_np[:min(depth, len(asks_np))]
                    bid_vol = np.sum(cb[:, 1]) if len(cb) > 0 else 0
                    ask_vol = np.sum(ca[:, 1]) if len(ca) > 0 else 0
                    total_vol = bid_vol + ask_vol
                    features[feat_name] = (bid_vol - ask_vol) / (total_vol + 1e-9)

            if 'price_impact_10' in self.l2_features_list_config:
                if len(bids_np) > 0 and len(asks_np) > 0:
                    mid_price = (bids_np[0, 0] + asks_np[0, 0]) / 2.0
                    if mid_price > 1e-9:
                        impact_depth_idx = self.config.get('l2_price_impact_depth_idx', 4)
                        if len(asks_np) > impact_depth_idx and len(bids_np) > impact_depth_idx:
                             features['price_impact_10'] = (asks_np[impact_depth_idx, 0] - bids_np[impact_depth_idx, 0]) / mid_price

            num_curve_levels = self.config.get('l2_curve_fit_levels', 20)
            if 'bid_curve' in self.l2_features_list_config:
                bp_c = bids_np[:min(num_curve_levels, len(bids_np)), 0]
                bv_c = bids_np[:min(num_curve_levels, len(bids_np)), 1]
                if len(bp_c) > 1: features['bid_curve'], _ = np.polyfit(bp_c, bv_c, 1)
            if 'ask_curve' in self.l2_features_list_config:
                ap_c = asks_np[:min(num_curve_levels, len(asks_np)), 0]
                av_c = asks_np[:min(num_curve_levels, len(asks_np)), 1]
                if len(ap_c) > 1: features['ask_curve'], _ = np.polyfit(ap_c, av_c, 1)
        except Exception as e:
            pass
        return features

    def _calculate_historical_l2_features(self, df_input):
        """Calculate historical L2 features (unchanged from original)."""
        df = df_input.copy()
        if not self.use_l2_features:
            for col in self.l2_features_list_config: df[col] = np.nan
            return df

        if 'l2_bids' not in df.columns or 'l2_asks' not in df.columns:
            print("Warning (FeatureEngineer): 'l2_bids' or 'l2_asks' columns not found for historical L2 features.")
            for col in self.l2_features_list_config: df[col] = np.nan
            return df

        l2_feature_rows = []
        for index, row in df.iterrows():
            bids = row['l2_bids']; asks = row['l2_asks']
            if isinstance(bids, list) and isinstance(asks, list):
                l2_feature_rows.append(self.calculate_l2_features_from_snapshot(bids, asks))
            else: l2_feature_rows.append({col: np.nan for col in self.l2_features_list_config})

        if l2_feature_rows:
            df_l2_features = pd.DataFrame(l2_feature_rows, index=df.index)
            for col in self.l2_features_list_config:
                if col in df_l2_features.columns: df[col] = df_l2_features[col]
                else: df[col] = np.nan
        else:
            for col in self.l2_features_list_config: df[col] = np.nan
        return df

    def generate_all_features(self, df_ohlcv_cleaned_aligned, save=True):
        """Generate all features with improved NaN handling to prevent dropping all rows."""
        if df_ohlcv_cleaned_aligned is None or df_ohlcv_cleaned_aligned.empty:
            print("Error (FeatureEngineer): Input DataFrame is empty.")
            return pd.DataFrame()

        print(f"Starting feature generation for {len(df_ohlcv_cleaned_aligned)} rows...")
        df = df_ohlcv_cleaned_aligned.copy()

        # Calculate features step by step
        df = self._calculate_zscore_features(df)
        df = self._calculate_spread_features(df)
        df = self._calculate_advanced_ta_features(df)
        
        if self.config.get('use_hht_features', False):
            df = self._calculate_hht_features(df)
        else:
            # Ensure HHT columns are NaN if not used
            hht_feature_names = [f'{base}{i}' for i in range(self.hht_imf_count) for base in self.hht_features_imf_bases]
            for col in hht_feature_names:
                if col not in df.columns: 
                    df[col] = np.nan

        if self.use_l2_features: 
            df = self._calculate_historical_l2_features(df)
        else:
            for col in self.l2_features_list_config:
                if col not in df.columns: 
                    df[col] = np.nan

        # More intelligent NaN handling - only drop rows if CORE features are missing
        # Core features are those absolutely essential for the model
        core_essential_features = []
        
        # Only include z-score features as core (these should rarely be NaN after our fixes)
        core_essential_features.extend([f for f in ['z_close', 'z_volume', 'z_spread'] if f in df.columns])
        
        initial_len = len(df)
        
        if core_essential_features:
            # First, check how many rows would be dropped
            nan_mask = df[core_essential_features].isnull().any(axis=1)
            rows_with_nans = nan_mask.sum()
            
            if rows_with_nans == len(df):
                print("Warning (FeatureEngineer): All rows would be dropped due to NaN in core features. Check feature calculations.")
                print("Core features causing issues:", core_essential_features)
                # Don't drop any rows in this case - let the model handle NaNs
            elif rows_with_nans > len(df) * 0.8:  # If more than 80% would be dropped
                print(f"Warning (FeatureEngineer): {rows_with_nans}/{len(df)} rows have NaN in core features. Keeping all rows.")
                # Don't drop rows - too much data loss
            else:
                # Safe to drop rows with NaN in core features
                df.dropna(subset=core_essential_features, how='any', inplace=True)
                rows_dropped = initial_len - len(df)
                if rows_dropped > 0: 
                    print(f"Dropped {rows_dropped} rows due to NaN in core features.")

        if df.empty:
            print("Error (FeatureEngineer): DataFrame empty after NaN handling.")
            return df

        # Fill remaining NaNs in TA features with forward fill, then 0
        ta_features_in_df = [f for f in self.ta_features_config if f in df.columns]
        for ta_feat in ta_features_in_df:
            df[ta_feat] = df[ta_feat].fillna(method='ffill').fillna(0.0)

        # Handle L2 features
        if self.use_l2_features and self.l2_features_list_config:
            actual_l2_cols_in_df = [col for col in self.l2_features_list_config if col in df.columns]
            if actual_l2_cols_in_df:
                l2_nan_counts = df[actual_l2_cols_in_df].isnull().sum()
                total_l2_nans = l2_nan_counts.sum()
                if total_l2_nans == len(df) * len(actual_l2_cols_in_df):
                    print("Warning (FeatureEngineer): L2 features enabled, but all values are NaN.")
                elif total_l2_nans > 0:
                    print(f"Warning (FeatureEngineer): {total_l2_nans} NaN values in L2 features. Filling with 0.")
                    # Fill L2 NaNs with 0 (neutral values for most L2 features)
                    for col in actual_l2_cols_in_df:
                        df[col] = df[col].fillna(0.0)
            else:
                print("Warning (FeatureEngineer): L2 features enabled in config, but no L2 feature columns in DataFrame.")

        # Final check - fill any remaining NaNs
        remaining_nans = df.isnull().sum().sum()
        if remaining_nans > 0:
            print(f"Warning (FeatureEngineer): {remaining_nans} NaN values remain. Filling with forward fill, then 0.")
            df = df.fillna(method='ffill').fillna(0.0)

        if save:
            try:
                os.makedirs(os.path.dirname(self.prepared_data_path), exist_ok=True)
                df.to_csv(self.prepared_data_path, index=False)
                print(f"Saved prepared data with features to {self.prepared_data_path}")
            except Exception as e:
                print(f"Warning (FeatureEngineer): Error saving prepared data: {e}")

        print(f"Feature generation complete. Final DataFrame shape: {df.shape}")
        return df