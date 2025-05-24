# smart_order_executor.py
# Reformatted from notebook export to standard Python file

import ccxt  # For type hinting and if specific ccxt exceptions are caught


class SmartOrderExecutor:
    """
    Handles order execution with considerations for slippage and order book dynamics.
    """

    def __init__(self, exchange_api, exec_config):
        """
        Initializes the SmartOrderExecutor.

        Args:
            exchange_api: An initialized CCXT exchange object.
            exec_config (dict): Dictionary containing execution parameters:
                - slippage_model_pct (float, optional): Percentage to adjust limit price for
                  better fill probability. Default 0.0005.
                - max_order_book_levels (int, optional): Max levels of order book to fetch for
                  walking. Default 20.
        """
        self.exchange = exchange_api
        self.slippage_model_pct = exec_config.get('slippage_model_pct', 0.0005)
        self.max_levels = exec_config.get('max_order_book_levels', 20)
        print("SmartOrderExecutor initialized.")

    def _walk_book(self, order_book_side_snapshot, amount_to_trade):
        """
        Simulates walking one side of the order book to estimate average fill price.

        Args:
            order_book_side_snapshot (list): List of [price, volume] for one side (bids or asks).
            amount_to_trade (float): The amount of asset to trade.

        Returns:
            tuple: (average_fill_price, filled_amount) or (None, 0) if not fillable.
        """
        filled_amount = 0.0
        total_cost_or_revenue = 0.0

        if not order_book_side_snapshot:
            return None, 0

        for price_level, volume_at_level in order_book_side_snapshot:
            if filled_amount >= amount_to_trade:
                break
            try:
                price = float(price_level)
                volume = float(volume_at_level)
                if price <= 0 or volume <= 0:
                    continue
            except (ValueError, TypeError):
                continue

            can_fill_this_level = min(volume, amount_to_trade - filled_amount)
            total_cost_or_revenue += can_fill_this_level * price
            filled_amount += can_fill_this_level

        if filled_amount == 0:
            return None, 0

        return total_cost_or_revenue / filled_amount, filled_amount

    def execute_order(self, symbol, side, amount, desired_price, order_type='limit'):
        """
        Executes a trading order.

        Args:
            symbol (str): The trading symbol (e.g., 'BTC/USDT').
            side (str): 'buy' or 'sell'.
            amount (float): The quantity of the asset to trade.
            desired_price (float): The desired price for a limit order (current price for market).
            order_type (str, optional): 'limit' or 'market'. Default 'limit'.

        Returns:
            dict or None: The order response from the exchange, or None if order fails.
        """
        if not self.exchange:
            print("Error (SmartExec): Exchange API not available.")
            return None

        if amount <= 1e-8:
            print(
                f"Warning (SmartExec): Order amount too small for {symbol}: {amount}. Skipping order."
            )
            return None

        final_limit_price = desired_price
        estimated_fill_price = desired_price

        try:
            if order_type == 'limit':
                if hasattr(self.exchange, 'fetch_l2_order_book') and \
                        self.exchange.has.get('fetchL2OrderBook'):
                    current_book = self.exchange.fetch_l2_order_book(
                        symbol, limit=self.max_levels
                    )
                    if current_book:
                        if side == 'buy' and current_book.get('asks') and current_book['asks']:
                            avg_price, _ = self._walk_book(current_book['asks'], amount)
                            if avg_price is not None:
                                estimated_fill_price = avg_price
                        elif side == 'sell' and current_book.get('bids') and current_book['bids']:
                            avg_price, _ = self._walk_book(current_book['bids'], amount)
                            if avg_price is not None:
                                estimated_fill_price = avg_price
                else:
                    print(
                        f"Warning (SmartExec): Exchange {self.exchange.id} does not support fetch_l2_order_book. "
                        "Using desired_price for limit order."
                    )

                if side == 'buy':
                    final_limit_price = estimated_fill_price * (1 + self.slippage_model_pct)
                elif side == 'sell':
                    final_limit_price = estimated_fill_price * (1 - self.slippage_model_pct)

        except Exception as e:
            print(
                f"Warning (SmartExec): Error during L2 book walk for {symbol}, "
                f"using desired_price for limit order: {e}"
            )
            final_limit_price = desired_price

        try:
            params = {}
            order_response = None

            log_price = (
                f"limit price {final_limit_price:.8f}" if order_type == 'limit'
                else f"approx. desired price {desired_price:.8f}"
            )
            print(
                f"Info (SmartExec): Attempting to place {order_type} {side} order for "
                f"{amount:.8f} of {symbol} at {log_price}"
            )

            if order_type == 'limit':
                order_response = self.exchange.create_order(
                    symbol=symbol, type='limit', side=side,
                    amount=amount, price=final_limit_price, params=params
                )
            elif order_type == 'market':
                order_response = self.exchange.create_order(
                    symbol=symbol, type='market', side=side,
                    amount=amount, params=params
                )
            else:
                print(f"Error (SmartExec): Unknown order type '{order_type}' for {symbol}.")
                return None

            if order_response and 'id' in order_response:
                print(
                    f"Info (SmartExec): Order placed successfully for {symbol}. "
                    f"Order ID: {order_response['id']}"
                )
            else:
                print(
                    f"Warning (SmartExec): Order placement for {symbol} did not return an ID or failed. "
                    f"Response: {order_response}"
                )
            return order_response

        except ccxt.NetworkError as e:
            print(f"Error (SmartExec) NetworkError placing order for {symbol}: {e}")
        except ccxt.ExchangeError as e:
            print(f"Error (SmartExec) ExchangeError placing order for {symbol}: {e}")
        except Exception as e:
            print(f"Error (SmartExec) Unexpected error placing order for {symbol}: {e}")

        return None