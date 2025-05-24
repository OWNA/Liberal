# advanced_risk_manager.py
# Reformatted from notebook export to standard Python file

import pandas as pd


class AdvancedRiskManager:
    """
    Manages risk through dynamic position sizing, volatility-adjusted stops,
    and take profit levels.
    """

    def __init__(self, risk_config):
        """
        Initializes the AdvancedRiskManager.

        Args:
            risk_config (dict): A dictionary containing risk management parameters:
                - max_drawdown (float, optional): Max acceptable drawdown.
                  Default 0.20.
                - volatility_lookback (int, optional): Lookback period for volatility.
                  Default 14.
                - position_sizing_mode (str, optional): 'volatility_target' or
                  'fixed_fraction'. Default 'volatility_target'.
                - volatility_target_pct (float, optional): Target volatility percentage.
                  Default 0.02.
                - max_equity_risk_pct (float, optional): Max percentage of equity to risk
                  per trade. Default 0.10.
                - fixed_fraction_pct (float, optional): Fixed fraction of equity for
                  position sizing. Default 0.05.
                - sl_atr_multiplier (float, optional): ATR multiplier for stop-loss.
                  Default 1.5.
                - tp_atr_multiplier (float, optional): ATR multiplier for take-profit.
                  Default 2.0.
        """
        self.max_drawdown = risk_config.get('max_drawdown', 0.20)
        self.volatility_lookback = risk_config.get('volatility_lookback', 14)
        self.position_sizing_mode = risk_config.get(
            'position_sizing_mode', 'volatility_target'
        )
        self.volatility_target_pct = risk_config.get('volatility_target_pct', 0.02)
        self.max_equity_risk_pct = risk_config.get('max_equity_risk_pct', 0.10)
        self.fixed_fraction_pct = risk_config.get('fixed_fraction_pct', 0.05)
        self.sl_atr_multiplier = risk_config.get('sl_atr_multiplier', 1.5)
        self.tp_atr_multiplier = risk_config.get('tp_atr_multiplier', 2.0)
        print("AdvancedRiskManager initialized.")

    def calculate_position_size(
        self, account_equity: float, current_volatility_pct: float
    ) -> float:
        """
        Calculates position size based on the configured mode.

        Args:
            account_equity (float): Current account equity.
            current_volatility_pct (float): Current price volatility percentage
                (e.g., ATR / price).

        Returns:
            float: Calculated position size in USD.
        """
        if self.position_sizing_mode == 'volatility_target':
            if current_volatility_pct <= 1e-5:
                # Fallback to a small fraction of max risk if volatility is negligible
                return account_equity * self.max_equity_risk_pct * 0.1

            size_usd = (
                account_equity * self.volatility_target_pct
            ) / current_volatility_pct
            max_size_usd = account_equity * self.max_equity_risk_pct
            return min(size_usd, max_size_usd)

        elif self.position_sizing_mode == 'fixed_fraction':
            return account_equity * self.fixed_fraction_pct
        else:
            # Default to fixed_fraction if mode is unknown
            print(
                f"Warning (RiskManager): Unknown position_sizing_mode "
                f"'{self.position_sizing_mode}'. Defaulting to 'fixed_fraction'."
            )
            return account_equity * self.fixed_fraction_pct

    def calculate_stop_loss(
        self, entry_price: float, atr_value: float, side: str = 'long'
    ) -> float:
        """
        Calculates stop-loss level.

        Args:
            entry_price (float): The entry price of the position.
            atr_value (float): The current Average True Range (ATR) value.
            side (str, optional): 'long' or 'short'. Default 'long'.

        Returns:
            float or None: Stop-loss price, or None if atr_value is invalid.
        """
        if not pd.notna(atr_value) or atr_value <= 0:
            return None

        if side == 'long':
            return entry_price - (atr_value * self.sl_atr_multiplier)
        elif side == 'short':
            return entry_price + (atr_value * self.sl_atr_multiplier)
        return None

    def calculate_take_profit(
        self, entry_price: float, atr_value: float, side: str = 'long'
    ) -> float:
        """
        Calculates take-profit level.

        Args:
            entry_price (float): The entry price of the position.
            atr_value (float): The current Average True Range (ATR) value.
            side (str, optional): 'long' or 'short'. Default 'long'.

        Returns:
            float or None: Take-profit price, or None if atr_value is invalid.
        """
        if not pd.notna(atr_value) or atr_value <= 0:
            return None

        if side == 'long':
            return entry_price + (atr_value * self.tp_atr_multiplier)
        elif side == 'short':
            return entry_price - (atr_value * self.tp_atr_multiplier)
        return None