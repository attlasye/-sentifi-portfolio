"""
Portfolio rebalancing logic
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple

class PortfolioRebalancer:
    """
    Handles portfolio rebalancing based on ML predictions
    """
    def __init__(self, 
                 rebalance_frequency: str = 'daily',
                 max_position_size: float = 0.30,
                 transaction_cost: float = 0.001):
        self.rebalance_frequency = rebalance_frequency
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        
    def calculate_target_weights(self, 
                               predictions: pd.DataFrame,
                               current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate target portfolio weights based on predictions
        """
        # Get latest predictions
        latest_pred = predictions.groupby('symbol').last()
        
        # Filter positive predictions
        positive_pred = latest_pred[latest_pred['ensemble_return_pred'] > 0]
        
        if len(positive_pred) == 0:
            return {}
            
        # Calculate weights proportional to expected returns
        expected_returns = positive_pred['ensemble_return_pred']
        raw_weights = expected_returns / expected_returns.sum()
        
        # Apply position limits
        weights = raw_weights.clip(upper=self.max_position_size)
        weights = weights / weights.sum()  # Renormalize
        
        return weights.to_dict()
        
    def rebalance_portfolio(self,
                          current_holdings: Dict[str, float],
                          target_weights: Dict[str, float],
                          portfolio_value: float) -> Tuple[Dict[str, float], float]:
        """
        Calculate trades needed to rebalance portfolio
        Returns: (trades, estimated_cost)
        """
        trades = {}
        total_cost = 0
        
        # Calculate current weights
        current_weights = {
            symbol: value / portfolio_value 
            for symbol, value in current_holdings.items()
        }
        
        # Calculate required trades
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        
        for symbol in all_symbols:
            current_weight = current_weights.get(symbol, 0)
            target_weight = target_weights.get(symbol, 0)
            
            weight_diff = target_weight - current_weight
            trade_value = weight_diff * portfolio_value
            
            if abs(trade_value) > portfolio_value * 0.001:  # Min trade size
                trades[symbol] = trade_value
                total_cost += abs(trade_value) * self.transaction_cost
                
        return trades, total_cost