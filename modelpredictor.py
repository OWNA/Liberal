# model_predictor.py
# Reformatted from notebook export to standard Python file

import os
import json
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb  # For loading lgb.Booster
import traceback  # For detailed error logging


class ModelPredictor:
    """
    Loads trained models, manages scaling information, and generates predictions/signals.
    """

    def __init__(self, config, data_handler=None, label_generator=None):
        """
        Initializes the ModelPredictor.

        Args:
            config (dict): Configuration dictionary.
            data_handler (DataHandler, optional): Instance of DataHandler, needed if
                scaling info needs to be recalculated.
            label_generator (LabelGenerator, optional): Instance of LabelGenerator,
                needed if scaling info needs to be recalculated.
        """
        self.config = config
        self.data_handler = data_handler
        self.label_generator = label_generator

        self.model_object = None
        self.trained_features = []

        self.target_mean = None
        self.target_std = None

        self.base_dir = config.get('base_dir', './trading_bot_data')
        safe_symbol = config.get('symbol', 'SYMBOL').replace('/', '_').replace(':', '')
        timeframe = config.get('timeframe', 'TIMEFRAME')

        self.model_path_default = os.path.join(
            self.base_dir,
            f"lgbm_model_{safe_symbol}_{timeframe}.txt"
        )
        self.ensemble_model_path_default = os.path.join(
            self.base_dir,
            f"ensemble_models_{safe_symbol}_{timeframe}.pkl"
        )
        self.features_json_path_default = os.path.join(
            self.base_dir,
            f"model_features_{safe_symbol}_{timeframe}.json"
        )

        print("ModelPredictor initialized.")

    def set_scaling_params(self, target_mean: float, target_std: float) -> None:
        """
        Sets the target scaling parameters externally.
        These should be the mean and std of the *scaled* target the model was trained on.
        """
        self.target_mean = target_mean
        self.target_std = target_std
        print(
            f"ModelPredictor: Scaling params set: mean={self.target_mean}, "
            f"std={self.target_std}"
        )

    def _ensure_scaling_info(self) -> bool:
        """
        Ensures target_mean and target_std are available.
        If not, attempts to recalculate them using DataHandler and LabelGenerator.
        This is a fallback and assumes historical data is available for recalculation.
        """
        if self.target_mean is not None and self.target_std is not None:
            return True

        print(
            "Warning (ModelPredictor): target_mean or target_std not set. "
            "Attempting to recalculate..."
        )
        if not self.data_handler or not self.label_generator:
            print(
                "Error (ModelPredictor): DataHandler or LabelGenerator not provided. "
                "Cannot recalculate scaling info."
            )
            self.target_mean = self.target_mean if self.target_mean is not None else 0.0
            self.target_std = self.target_std if self.target_std is not None else 1.0
            print(
                f"ModelPredictor: Using fallback scaling: mean={self.target_mean}, "
                f"std={self.target_std}"
            )
            return False

        try:
            print(
                "ModelPredictor: Attempting to load historical data for scaling info "
                "recalculation."
            )
            df_for_scaling_raw = self.data_handler.load_and_prepare_historical_data(
                fetch_ohlcv_limit=self.config.get('fetch_ohlcv_limit_for_scaling', 500),
                use_historical_l2=False,
                save_ohlcv=False
            )

            if (
                df_for_scaling_raw is None
                or df_for_scaling_raw.empty
                or 'close' not in df_for_scaling_raw.columns
            ):
                print(
                    "Error (ModelPredictor): Failed to get 'close' data for recalculating "
                    "scaling info."
                )
                self.target_mean = 0.0
                self.target_std = 1.0
                return False

            _, mean_val, std_val = self.label_generator.generate_labels(df_for_scaling_raw)

            if mean_val is not None and std_val is not None:
                self.target_mean = mean_val
                self.target_std = std_val
                print(
                    f"ModelPredictor: Recalculated scaling info: mean={self.target_mean}, "
                    f"std={self.target_std}"
                )
                return True
            else:
                print(
                    "Error (ModelPredictor): Failed to recalculate scaling info from "
                    "LabelGenerator."
                )
                self.target_mean = 0.0
                self.target_std = 1.0
                return False

        except Exception as e:
            print(f"Error (ModelPredictor): Exception during scaling info recalculation: {e}")
            traceback.print_exc()
            self.target_mean = self.target_mean if self.target_mean is not None else 0.0
            self.target_std = self.target_std if self.target_std is not None else 1.0
            return False

    def load_model_and_features(
        self,
        model_file_path: str = None,
        features_file_path: str = None,
        load_ensemble: bool = False
    ) -> bool:
        """
        Loads a trained model (standard booster or ensemble) and its feature list.
        Also ensures scaling information is available.
        """
        _model_path = model_file_path or (
            self.ensemble_model_path_default if load_ensemble else self.model_path_default
        )
        _features_path = features_file_path

        try:
            if load_ensemble:
                with open(_model_path, 'rb') as f:
                    ensemble_content = pickle.load(f)
                if isinstance(ensemble_content, dict):
                    self.model_object = ensemble_content
                    self.trained_features = ensemble_content.get('trained_features', [])
                    self.target_mean = ensemble_content.get('target_mean', self.target_mean)
                    self.target_std = ensemble_content.get('target_std', self.target_std)

                    if not self.trained_features and _features_path and os.path.exists(_features_path):
                        with open(_features_path, 'r') as f_json:
                            self.trained_features = json.load(f_json)
                else:
                    print(
                        f"Error (ModelPredictor): Ensemble model at {_model_path} is not a dictionary."
                    )
                    return False
                print(f"Ensemble model loaded from {_model_path}")
            else:
                self.model_object = lgb.Booster(model_file=_model_path)
                _features_path_to_load = _features_path or self.features_json_path_default
                if os.path.exists(_features_path_to_load):
                    with open(_features_path_to_load, 'r') as f_json:
                        self.trained_features = json.load(f_json)
                else:
                    print(
                        f"Warning (ModelPredictor): Features JSON file not found at "
                        f"{_features_path_to_load} for standard model."
                    )
                print(f"Standard model booster loaded from {_model_path}")

            if not self.trained_features:
                print(
                    f"Warning (ModelPredictor): No trained features loaded from path or ensemble "
                    f"for model '{_model_path}'."
                )

            print(
                f"ModelPredictor: Loaded {len(self.trained_features)} features: "
                f"{self.trained_features[:5]}..."
            )

            if not self._ensure_scaling_info():
                print(
                    "Warning (ModelPredictor): Scaling info could not be confirmed/recalculated. "
                    "Predictions might be unscaled or use defaults."
                )
            return True

        except FileNotFoundError:
            print(
                f"Error (ModelPredictor): Model or features file not found. Model: '{_model_path}', "
                f"Features path used: '{_features_path if _features_path else 'within ensemble or default'}'"
            )
            self.model_object = None
            self.trained_features = []
            return False
        except Exception as e:
            print(f"Error (ModelPredictor): Loading model/features failed: {e}")
            traceback.print_exc()
            self.model_object = None
            self.trained_features = []
            return False

    def predict_signals(
        self,
        df_with_features: pd.DataFrame,
        threshold: float = None,
        use_ensemble: bool = False
    ) -> pd.DataFrame:
        """
        Generates predictions and trading signals on the input DataFrame.
        """
        if self.model_object is None:
            print(
                "Error (ModelPredictor): Model not loaded. Call load_model_and_features() first."
            )
            return None
        if not self.trained_features:
            print(
                "Error (ModelPredictor): Trained feature list not available. "
                "Cannot determine input features for model."
            )
            return None
        if df_with_features is None or df_with_features.empty:
            print("Error (ModelPredictor): Input DataFrame for prediction is empty.")
            return None

        _threshold = threshold if threshold is not None else self.config.get(
            'prediction_threshold', self.config.get('backtest_threshold', 0.5)
        )

        if self.target_mean is None or self.target_std is None:
            print(
                "Warning (ModelPredictor): Scaling parameters (target_mean, target_std) are not set. "
                "Unscaling will not be performed or use defaults."
            )
            if not self._ensure_scaling_info():
                print(
                    "Critical Warning (ModelPredictor): Failed to ensure scaling info. "
                    "Predictions might be inaccurate if model expects scaled target."
                )

        missing_features = [
            f for f in self.trained_features if f not in df_with_features.columns
        ]
        if missing_features:
            print(
                f"Error (ModelPredictor): Input DataFrame is missing required features: "
                f"{missing_features}"
            )
            return None

        X_predict = df_with_features[self.trained_features].copy()

        if X_predict.isnull().values.any():
            nan_cols = X_predict.columns[X_predict.isnull().any()].tolist()
            print(
                f"Warning (ModelPredictor): NaNs found in features for prediction: "
                f"{nan_cols}. Model might handle them if trained accordingly. Consider imputation."
            )

        result_df = df_with_features.copy()

        try:
            if use_ensemble:
                if not isinstance(self.model_object, dict) or 'classifier' not in self.model_object:
                    print(
                        "Error (ModelPredictor): Ensemble model not loaded correctly or 'classifier' missing."
                    )
                    return None

                clf = self.model_object['classifier']
                reg = self.model_object.get('regressor')
                clf_map_to_lgbm = self.model_object.get('clf_target_map_to_lgbm', {-1: 0, 0: 1, 1: 2})
                clf_map_from_lgbm = {v: k for k, v in clf_map_to_lgbm.items()}

                clf_preds_raw = clf.predict(X_predict)
                result_df["signal"] = pd.Series(
                    clf_preds_raw, index=X_predict.index
                ).map(clf_map_from_lgbm).fillna(0).astype(int)

                if reg:
                    pred_reg_scaled = reg.predict(X_predict)
                    result_df["pred_scaled"] = pred_reg_scaled
                    if (
                        self.target_mean is not None
                        and self.target_std is not None
                        and self.target_std > 1e-9
                    ):
                        result_df["pred_unscaled_target"] = (
                            pred_reg_scaled * self.target_std
                        ) + self.target_mean
                    else:
                        result_df["pred_unscaled_target"] = pred_reg_scaled
                else:
                    result_df["pred_scaled"] = result_df["signal"] * _threshold
                    result_df["pred_unscaled_target"] = np.nan
            else:
                pred_scaled = self.model_object.predict(X_predict)
                result_df["pred_scaled"] = pred_scaled

                if (
                    self.target_mean is not None
                    and self.target_std is not None
                    and self.target_std > 1e-9
                ):
                    result_df["pred_unscaled_target"] = (
                        pred_scaled * self.target_std
                    ) + self.target_mean
                else:
                    result_df["pred_unscaled_target"] = pred_scaled

                result_df["signal"] = np.select(
                    [pred_scaled > _threshold, pred_scaled < -_threshold],
                    [1, -1],
                    default=0
                )
            print("Predictions and signals generated.")
        except Exception as e:
            print(f"Error (ModelPredictor): Prediction failed: {e}")
            traceback.print_exc()
            return None

        return result_df