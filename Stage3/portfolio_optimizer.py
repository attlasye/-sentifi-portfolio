# portfolio_optimizer.py
# FINAL VERSION with enhanced methods and proper formatting

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional, List
import cvxpy as cp
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self,
                 max_position: float = 0.10,
                 target_vol: float = 0.40,
                 lookback_days: int = 60,
                 rebalance_threshold: float = 0.05):
        self.max_position = max_position
        self.target_vol = target_vol
        self.lookback_days = lookback_days
        self.rebalance_threshold = rebalance_threshold
        
    def mean_variance_optimization(self,
                                 expected_returns: pd.Series,
                                 covariance_matrix: pd.DataFrame,
                                 risk_aversion: float = 1.0) -> pd.Series:
        n_assets = len(expected_returns)
        weights = cp.Variable(n_assets)
        portfolio_return = expected_returns.values @ weights
        portfolio_variance = cp.quad_form(weights, covariance_matrix.values)
        objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0,
            weights <= self.max_position,
        ]
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
        except Exception as e:
            logger.error(f"MV optimization failed: {e}")
            return pd.Series(1/n_assets, index=expected_returns.index)

        if problem.status != 'optimal':
            logger.warning(f"MV optimization status: {problem.status}")
            return pd.Series(1/n_assets, index=expected_returns.index)
        
        return pd.Series(weights.value, index=expected_returns.index)
    
    def risk_parity_optimization(self,
                               covariance_matrix: pd.DataFrame,
                               expected_returns: Optional[pd.Series] = None,
                               alpha: float = 0.7) -> pd.Series:
        """
        Risk parity optimization with optional blending of ML signals.
        """
        n_assets = len(covariance_matrix)
        
        def risk_contribution(weights, cov_matrix):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_vol
            return contrib
        
        def objective(weights):
            contrib = risk_contribution(weights, covariance_matrix.values)
            target_contrib = np.ones(n_assets) / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, self.max_position) for _ in range(n_assets)]
        w0 = np.ones(n_assets) / n_assets
        
        result = minimize(objective, w0, 
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints,
                         options={'maxiter': 1000})
        
        if not result.success:
            logger.warning(f"Pure RP optimization failed: {result.message}")
            rp_weights = pd.Series(1/n_assets, index=covariance_matrix.index)
        else:
            rp_weights = pd.Series(result.x, index=covariance_matrix.index)
            
        if expected_returns is not None and not expected_returns.empty:
            valid_rp_weights = rp_weights.loc[expected_returns.index]
            ml_weights = expected_returns.clip(lower=0)
            if ml_weights.sum() > 1e-8:
                ml_weights = ml_weights / ml_weights.sum()
            else:
                ml_weights = pd.Series(1/len(expected_returns), index=expected_returns.index)

            final_weights = alpha * valid_rp_weights + (1 - alpha) * ml_weights
            final_weights = final_weights / final_weights.sum()
            
            return final_weights.clip(upper=self.max_position)
            
        return rp_weights.clip(upper=self.max_position)
    
    def apply_volatility_target(self,
                              weights: pd.Series,
                              returns_df: pd.DataFrame) -> Tuple[pd.Series, float]:
        portfolio_returns = returns_df @ weights
        current_vol = portfolio_returns.std() * np.sqrt(252)
        vol_scalar = self.target_vol / current_vol
        vol_scalar = min(vol_scalar, 1.0)
        
        adjusted_weights = weights * vol_scalar
        cash_weight = 1 - adjusted_weights.sum()
        
        logger.info(f"Vol adjustment: current={current_vol:.1%}, target={self.target_vol:.1%}, scalar={vol_scalar:.2f}")
        
        return adjusted_weights, cash_weight