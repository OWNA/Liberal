# model_trainer.py
# Reformatted from notebook export to standard Python file

import os
import json
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import traceback

# optuna will be an optional import, checked by HAS_OPTUNA flag

class ModelTrainer:
    """
    Handles the training of LightGBM models (standard and ensemble)
    and manages model/feature persistence.
    """
    def __init__(self, config, feature_list_all_defined, has_optuna=False, optuna_module=None):
        """
        Initializes the ModelTrainer.
        (Constructor arguments remain the same)
        """
        self.config = config
        self.feature_list_all_defined = feature_list_all_defined
        self.trained_features = []

        self.HAS_OPTUNA = has_optuna
        self.optuna = optuna_module

        # --- Phase 1a: Optuna search space configuration ---
        self.optuna_search_spaces = config.get('optuna_search_spaces', {}) # Load custom search spaces

        self.base_dir = config.get('base_dir', './trading_bot_data')
        safe_symbol = config.get('symbol', 'SYMBOL').replace('/', '_').replace(':', '')
        timeframe = config.get('timeframe', 'TIMEFRAME')

        self.model_path = os.path.join(self.base_dir, f"lgbm_model_{safe_symbol}_{timeframe}.txt")
        self.ensemble_model_path = os.path.join(self.base_dir, f"ensemble_models_{safe_symbol}_{timeframe}.pkl")
        self.features_json_path = os.path.join(self.base_dir, f"model_features_{safe_symbol}_{timeframe}.json") # For standard model
        # For ensemble, features are now saved within the .pkl file.

        self.l2_features_list_config = config.get('l2_features', [])
        self.use_l2_features_in_model = config.get('use_l2_features', False)

        print("ModelTrainer initialized (Phase 1 Update).")

    def _prepare_training_data(self, df_labeled_features_input):
        """
        Prepares X (features) and y (target) for training.
        Dynamically determines actual features to use by:
        1. Considering only features from `self.feature_list_all_defined`.
        2. Ensuring these features exist in the input DataFrame.
        3. Ensuring these features are not entirely NaN.
        4. Handling L2 features based on `use_l2_features_in_model` and whether they are all NaN.
        5. Dropping rows with any NaNs in the final selected features or target(s).
        """
        if df_labeled_features_input is None or df_labeled_features_input.empty:
            print("Error (ModelTrainer): Labeled DataFrame is empty for training data preparation.")
            return None, None, []

        df_train_ready = df_labeled_features_input.copy()

        if "target" not in df_train_ready.columns:
            print("Error (ModelTrainer): 'target' column not in DataFrame for training.")
            return None, None, []

        potential_features = [f for f in self.feature_list_all_defined if f in df_train_ready.columns]

        current_trained_features = []
        l2_cols_in_data = [f for f in self.l2_features_list_config if f in potential_features]
        l2_all_nan_in_data = False

        if self.use_l2_features_in_model and l2_cols_in_data:
            if df_train_ready[l2_cols_in_data].isnull().all().all():
                l2_all_nan_in_data = True
                print("Warning (ModelTrainer): L2 features are configured for use, but all L2 columns in the provided data are NaN. These L2 features will be excluded from training.")

        for f_name in potential_features:
            if df_train_ready[f_name].isnull().all():
                # print(f"Info (ModelTrainer): Feature '{f_name}' is entirely NaN, excluding from training.")
                continue # Skip features that are all NaN

            is_l2_feature = f_name in self.l2_features_list_config
            if self.use_l2_features_in_model and is_l2_feature and l2_all_nan_in_data:
                # Already logged above, so just skip
                continue

            current_trained_features.append(f_name)

        if not current_trained_features:
            print("Error (ModelTrainer): No valid (non-all-NaN) features found from the defined list in the provided data.")
            return None, None, []

        # --- Placeholder for Phase 1c: Automated Feature Selection ---
        # if self.config.get('enable_feature_selection', False):
        #     print("Info (ModelTrainer): Automated feature selection is enabled.")
        #     # selected_features = self._perform_feature_selection(df_train_ready[current_trained_features], df_train_ready['target'])
        #     # if selected_features:
        #     #     current_trained_features = selected_features
        #     # else:
        #     #     print("Warning (ModelTrainer): Feature selection did not return any features. Using original set.")
        #     pass # Implement _perform_feature_selection method based on config (SHAP, RFE, L1 etc.)

        # Drop rows with NaNs in the final selected features or target(s)
        columns_to_check_for_nan = current_trained_features + ['target']
        if 'clf_target' in df_train_ready.columns: # For ensemble model
             columns_to_check_for_nan.append('clf_target')

        initial_rows = len(df_train_ready)
        df_train_ready.dropna(subset=columns_to_check_for_nan, inplace=True)
        rows_dropped = initial_rows - len(df_train_ready)
        if rows_dropped > 0:
            print(f"Info (ModelTrainer): Dropped {rows_dropped} rows due to NaNs in selected features or target(s).")


        if df_train_ready.empty:
            print("Error (ModelTrainer): DataFrame is empty after NaN checks based on selected features and target(s).")
            return None, None, []

        X = df_train_ready[current_trained_features]
        y = df_train_ready["target"] # This is the regression target

        min_samples = self.config.get('min_training_samples', 50)
        if len(X) < min_samples:
            print(f"Error (ModelTrainer): Insufficient data ({len(X)} samples, need {min_samples}) for training after preparation.")
            return None, None, []

        return X, y, current_trained_features



    def train_standard_model(self, df_labeled_features, save=True):
        """
        Trains a standard LightGBM regression model with Optuna hyperparameter optimization.
        """
        if not self.HAS_OPTUNA or not self.optuna:
            print("Error (ModelTrainer): Optuna not available, cannot train standard model.")
            return None, []

        print("Starting standard model training...")
        X, y_reg, actual_features_used = self._prepare_training_data(df_labeled_features)

        if X is None or y_reg is None or not actual_features_used:
            print("Error (ModelTrainer): Data preparation for standard model training failed.")
            return None, []

        self.trained_features = actual_features_used
        print(f"Training standard model with {len(self.trained_features)} features: {self.trained_features[:10]}...") # Print first 10

        X_train, X_val, y_train, y_val = train_test_split(
            X, y_reg, test_size=self.config.get('test_size', 0.2),
            shuffle=self.config.get('train_test_split_shuffle', False), # Configurable shuffle
            random_state=self.config.get('random_state', 42) if self.config.get('train_test_split_shuffle', False) else None
        )

        def objective(trial):
            # --- Phase 1a: Use configurable Optuna search spaces ---
            params = {
                'objective': 'regression_l1',
                'metric': 'mae',
                'verbosity': -1,
                'random_state': self.config.get('random_state', 42),
                'n_jobs': self.config.get('lgbm_n_jobs', -1) # Use all available cores by default
            }

            # Default search spaces (can be overridden by config)
            default_search_spaces = {
                'n_estimators': {'type': 'int', 'low': 100, 'high': self.config.get('optuna_n_estimators_max', 1000), 'step': 50},
                'learning_rate': {'type': 'float', 'low': 1e-3, 'high': 0.2, 'log': True},
                'num_leaves': {'type': 'int', 'low': 20, 'high': 150},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'lambda_l1': {'type': 'float', 'low': 1e-7, 'high': 5.0, 'log': True},
                'lambda_l2': {'type': 'float', 'low': 1e-7, 'high': 5.0, 'log': True},
                'feature_fraction': {'type': 'float', 'low': 0.5, 'high': 1.0}, # Reduced lower bound
                'bagging_fraction': {'type': 'float', 'low': 0.5, 'high': 1.0}, # Reduced lower bound
                'bagging_freq': {'type': 'int', 'low': 1, 'high': 7},
                'min_child_samples': {'type': 'int', 'low': 5, 'high': 50} # Reduced bounds
            }

            for param_name, space_config in default_search_spaces.items():
                # Allow override from self.optuna_search_spaces in config.yaml
                config_space = self.optuna_search_spaces.get(param_name, space_config)

                param_type = config_space.get('type', 'float') # Default to float if not specified
                low = config_space['low']
                high = config_space['high']

                if param_type == 'int':
                    step = config_space.get('step', 1)
                    params[param_name] = trial.suggest_int(param_name, low, high, step=step)
                elif param_type == 'float':
                    log = config_space.get('log', False)
                    step = config_space.get('step') # Optional step for float
                    if step:
                        params[param_name] = trial.suggest_float(param_name, low, high, step=step, log=log)
                    else:
                        params[param_name] = trial.suggest_float(param_name, low, high, log=log)
                elif param_type == 'categorical':
                    choices = config_space['choices']
                    params[param_name] = trial.suggest_categorical(param_name, choices)

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mae',\
                        callbacks=[lgb.early_stopping(self.config.get('optuna_early_stopping_rounds', 25), verbose=False)]) # Increased early stopping
            return mean_absolute_error(y_val, model.predict(X_val))

        study_name = self.config.get('optuna_study_name', f'lgbm_optimization_{self.config.get("symbol","")}_{self.config.get("timeframe","")}')
        study = self.optuna.create_study(\
            study_name=study_name,\
            direction='minimize',\
            load_if_exists=self.config.get('optuna_load_if_exists', True) # Allow resuming studies
        )

        n_trials = self.config.get('optuna_trials', 30)
        timeout_seconds = self.config.get('optuna_timeout_seconds') # Optional timeout

        study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds,\
                       n_jobs=self.config.get('optuna_n_jobs', 1)) # Allow parallel trials if optuna storage is set up

        final_params = study.best_params
        final_params.update({'objective': 'regression_l1', 'metric':'mae', 'verbosity':-1,\
                             'random_state':self.config.get('random_state', 42),\
                             'n_jobs': self.config.get('lgbm_n_jobs', -1)})

        print(f"Best MAE from Optuna: {study.best_value:.6f}")
        print(f"Best Optuna params: {final_params}")

        print("Training final standard model on all available data (X, y_reg)...")
        final_model_regressor = lgb.LGBMRegressor(**final_params)
        final_model_regressor.fit(X, y_reg)

        lgbm_booster = final_model_regressor.booster_

        if save:
            try:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                lgbm_booster.save_model(self.model_path)
                print(f"Standard model saved to {self.model_path}")

                # Save features used for this specific model
                features_to_save = {
                    'trained_features': self.trained_features,
                    'target_mean': self.config.get('target_mean_for_prediction'), # Get from orchestrator if set
                    'target_std': self.config.get('target_std_for_prediction')   # Get from orchestrator if set
                }
                with open(self.features_json_path, 'w') as f:
                    json.dump(features_to_save, f, indent=4)
                print(f"Feature list and scaling info for standard model saved to {self.features_json_path}")

            except Exception as e:
                print(f"Warning (ModelTrainer): Error saving standard model or features: {e}")
                traceback.print_exc()

        print("Standard model training complete.")
        return lgbm_booster, self.trained_features

    def train_ensemble_model(self, df_labeled_features, save=True):
        """
        Trains an ensemble of LightGBM classifier and regressor.
        """
        print("Starting ensemble model training...")
        df_train_temp = df_labeled_features.copy()

        long_thresh = self.config.get('ensemble_long_thresh', 0.5)
        short_thresh = self.config.get('ensemble_short_thresh', -0.5)

        if "target" not in df_train_temp.columns:
             print("Error (ModelTrainer): 'target' column missing for ensemble training.")
             return None, []

        df_train_temp['clf_target'] = 0 # Neutral
        df_train_temp.loc[df_train_temp['target'] > long_thresh, 'clf_target'] = 1  # Long
        df_train_temp.loc[df_train_temp['target'] < short_thresh, 'clf_target'] = -1 # Short
        df_train_temp['clf_target'] = df_train_temp['clf_target'].astype(int)

        X, y_reg, actual_features_used = self._prepare_training_data(df_train_temp)

        if X is None or y_reg is None or not actual_features_used:
            print("Error (ModelTrainer): Data preparation for ensemble model training failed.")
            return None, []

        self.trained_features = actual_features_used
        print(f"Training ensemble model with {len(self.trained_features)} features: {self.trained_features[:10]}...")

        y_clf = df_train_temp.loc[X.index, 'clf_target'] # Align clf_target with X after NaN drops from _prepare_training_data

        clf_params = self.config.get('ensemble_clf_params', {
            'objective': 'multiclass', 'metric': 'multi_logloss',
            'num_class': 3, 'n_estimators': 150,
            'random_state': self.config.get('random_state', 42), 'verbosity': -1,
            'n_jobs': self.config.get('lgbm_n_jobs', -1)
        })
        # Ensure all necessary default params are present if not in config
        clf_params.setdefault('random_state', self.config.get('random_state', 42))
        clf_params.setdefault('verbosity', -1)
        clf_params.setdefault('n_jobs', self.config.get('lgbm_n_jobs', -1))



        clf_target_map_to_lgbm = {-1: 0, 0: 1, 1: 2} # Map: Trading Signal -> LGBM Class
        y_clf_mapped = y_clf.map(clf_target_map_to_lgbm)

        # Check if all 3 classes are present for multiclass objective
        if len(y_clf_mapped.unique()) < 3 and clf_params.get('objective') == 'multiclass':
            print(f"Warning (ModelTrainer): Only {len(y_clf_mapped.unique())} unique classes found for classifier target, but objective is 'multiclass' with num_class=3. Model might not train well or error.")
            # Optionally, could switch to binary if only two classes, or adjust num_class, but this indicates data/threshold issues.

        classifier = lgb.LGBMClassifier(**clf_params)
        classifier.fit(X, y_clf_mapped)
        print("Ensemble classifier trained.")

        reg_params = self.config.get('ensemble_reg_params', {
            'objective': 'regression_l1', 'metric': 'mae',
            'n_estimators': 150,
            'random_state': self.config.get('random_state', 42), 'verbosity': -1,
            'n_jobs': self.config.get('lgbm_n_jobs', -1)
        })
        reg_params.setdefault('random_state', self.config.get('random_state', 42))
        reg_params.setdefault('verbosity', -1)
        reg_params.setdefault('n_jobs', self.config.get('lgbm_n_jobs', -1))

        regressor = lgb.LGBMRegressor(**reg_params)
        regressor.fit(X, y_reg)
        print("Ensemble regressor trained.")

        ensemble_models = {
            'classifier': classifier,
            'regressor': regressor,
            'clf_target_map_to_lgbm': clf_target_map_to_lgbm,
            'trained_features': self.trained_features,
            'target_mean': self.config.get('target_mean_for_prediction'), # Get from orchestrator
            'target_std': self.config.get('target_std_for_prediction')    # Get from orchestrator
        }

        if save:
            try:
                os.makedirs(os.path.dirname(self.ensemble_model_path), exist_ok=True)
                with open(self.ensemble_model_path, 'wb') as f:
                    pickle.dump(ensemble_models, f)
                print(f"Ensemble model (including features and scaling info) saved to {self.ensemble_model_path}")
            except Exception as e:
                print(f"Warning (ModelTrainer): Error saving ensemble model: {e}")
                traceback.print_exc()

        print("Ensemble model training complete.")
        return ensemble_models, self.trained_features