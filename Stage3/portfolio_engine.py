# Stage3/portfolio_engine.py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

# Import existing Stage3 modules
from portfolio_optimizer import PortfolioOptimizer
from integrated_pipeline import integrate_features
from crypto_pipeline import stage2_feature_engineering
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PortfolioEngine:
    """
    Wrapper class that uses Stage3 optimization tools for real-time portfolio optimization
    """
    
    def __init__(self):
        self.optimizer = PortfolioOptimizer(
            max_position=0.40,
            target_vol=0.40,
            lookback_days=60
        )
        self.data_cache = {}
        self.last_update = None
        self._load_latest_data()
    
    def _load_latest_data(self):
        """Load the most recent data from Stage3 results"""
        try:
            # Load pre-computed features
            features_path = Path("./results/integrated_results/integrated_features.csv")
            if features_path.exists():
                self.features_df = pd.read_csv(features_path)
                self.features_df['date'] = pd.to_datetime(self.features_df['date'])
                
            # Load sentiment data
            sentiment_path = Path("./results/news_data/compound_timeseries_daily.csv")
            if sentiment_path.exists():
                self.sentiment_df = pd.read_csv(sentiment_path)
                self.sentiment_df['date'] = pd.to_datetime(self.sentiment_df['date'])
                
            # Load ML predictions
            ml_path = Path("./results/integrated_results/ml_predictions_integrated.csv")
            if ml_path.exists():
                self.ml_predictions = pd.read_csv(ml_path)
                self.ml_predictions['date'] = pd.to_datetime(self.ml_predictions['date'])
                
            self.last_update = datetime.now()
            logger.info("Data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def optimize(
        self,
        assets: List[str],
        objective: str = "max_sharpe",
        constraints: Optional[Dict] = None,
        use_sentiment: bool = True
    ) -> Dict:
        """
        Perform portfolio optimization for given assets
        """
        
        # Filter data for requested assets
        asset_data = self._prepare_asset_data(assets)
        
        if asset_data.empty:
            raise ValueError(f"No data available for assets: {assets}")
        
        # Get expected returns (from ML predictions or historical)
        expected_returns = self._get_expected_returns(assets, use_sentiment)
        
        # Calculate covariance matrix
        returns_matrix = self._get_returns_matrix(assets)
        cov_matrix = returns_matrix.cov()
        
        # Perform optimization based on objective
        if objective == "max_sharpe":
            weights = self.optimizer.mean_variance_optimization(
                expected_returns=expected_returns,
                covariance_matrix=cov_matrix,
                risk_aversion=1.0
            )
        elif objective == "min_risk":
            weights = self.optimizer.mean_variance_optimization(
                expected_returns=expected_returns,
                covariance_matrix=cov_matrix,
                risk_aversion=10.0  # High risk aversion for min risk
            )
        elif objective == "risk_parity":
            weights = self.optimizer.risk_parity_optimization(
                covariance_matrix=cov_matrix,
                expected_returns=expected_returns if use_sentiment else None
            )
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Apply constraints
        if constraints:
            weights = self._apply_constraints(weights, constraints)
        
        # Calculate portfolio metrics
        metrics = self._calculate_metrics(weights, expected_returns, cov_matrix)
        
        return {
            "weights": weights.to_dict(),
            "metrics": metrics,
            "optimization_method": objective,
            "assets_used": assets
        }
    
    def _prepare_asset_data(self, assets: List[str]) -> pd.DataFrame:
        """Filter features dataframe for requested assets"""
        if hasattr(self, 'features_df'):
            return self.features_df[self.features_df['symbol'].isin(assets)]
        return pd.DataFrame()
    
    def _get_expected_returns(
        self, 
        assets: List[str], 
        use_sentiment: bool
    ) -> pd.Series:
        """
        Get expected returns from ML predictions or calculate from historical data
        """
        
        if hasattr(self, 'ml_predictions') and use_sentiment:
            # Use ML predictions
            latest_predictions = self.ml_predictions.groupby('symbol').last()
            predictions = latest_predictions.loc[
                latest_predictions.index.isin(assets), 
                'ensemble_return_pred'
            ]
            return predictions
        
        # Fallback to historical average returns
        returns_matrix = self._get_returns_matrix(assets)
        return returns_matrix.mean()
    
    def _get_returns_matrix(self, assets: List[str]) -> pd.DataFrame:
        """Get returns matrix for covariance calculation"""
        if hasattr(self, 'features_df'):
            df = self.features_df[self.features_df['symbol'].isin(assets)]
            pivot = df.pivot(index='date', columns='symbol', values='return')
            return pivot.dropna()
        
        # Generate dummy data if no real data available
        dates = pd.date_range(end=datetime.now(), periods=100)
        returns = pd.DataFrame(
            np.random.randn(100, len(assets)) * 0.05,
            index=dates,
            columns=assets
        )
        return returns
    
    def _apply_constraints(
        self, 
        weights: pd.Series, 
        constraints: Dict
    ) -> pd.Series:
        """Apply weight constraints"""
        
        max_weight = constraints.get('max_weight', 1.0)
        min_weight = constraints.get('min_weight', 0.0)
        
        # Clip weights
        weights = weights.clip(lower=min_weight, upper=max_weight)
        
        # Renormalize
        weights = weights / weights.sum()
        
        return weights
    
    def _calculate_metrics(
        self, 
        weights: pd.Series,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> Dict:
        """Calculate portfolio metrics"""
        
        # Portfolio return
        portfolio_return = (weights * expected_returns).sum()
        
        # Portfolio volatility
        portfolio_variance = weights @ cov_matrix @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            "expected_return": float(portfolio_return * 365),  # Annualized
            "volatility": float(portfolio_volatility * np.sqrt(365)),  # Annualized
            "sharpe_ratio": float(sharpe_ratio * np.sqrt(365)),  # Annualized
            "max_weight": float(weights.max()),
            "min_weight": float(weights.min()),
            "num_assets": len(weights[weights > 0.01])  # Count significant positions
        }
    
    def backtest(
        self,
        assets: List[str],
        weights: Dict[str, float],
        start_date: str,
        end_date: str
    ) -> Dict:
        """
        Backtest a portfolio configuration
        """
        
        # Convert weights dict to Series
        weights_series = pd.Series(weights)
        
        # Get historical returns
        returns_matrix = self._get_returns_matrix(assets)
        
        # Filter by date range
        returns_matrix = returns_matrix[start_date:end_date]
        
        # Calculate portfolio returns
        portfolio_returns = (returns_matrix * weights_series).sum(axis=1)
        
        # Calculate metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = (1 + total_return) ** (365 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(365)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            "returns": portfolio_returns.tolist(),
            "metrics": {
                "total_return": float(total_return),
                "annual_return": float(annual_return),
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe),
                "max_drawdown": float(max_drawdown)
            },
            "benchmark_comparison": {
                "vs_bitcoin": float(annual_return - 0.5),  # Placeholder
                "vs_equal_weight": float(annual_return - 0.3)  # Placeholder
            }
        }
    
    def get_latest_sentiment(self) -> Dict:
        """Get latest sentiment scores"""
        
        if hasattr(self, 'sentiment_df'):
            latest = self.sentiment_df.iloc[-1]
            
            # Calculate trend (last 7 days)
            recent = self.sentiment_df.tail(7)
            trend = "bullish" if recent['Overall'].mean() > 60 else \
                    "bearish" if recent['Overall'].mean() < 40 else "neutral"
            
            return {
                "overall": float(latest.get('Overall', 50)),
                "by_asset": {
                    col: float(latest.get(col, 50))
                    for col in ['BTC', 'ETH', 'SOL', 'ADA']
                    if col in latest
                },
                "trend": trend
            }
        
        return {
            "overall": 50.0,
            "by_asset": {},
            "trend": "neutral"
        }
    
    def check_rebalance_needed(
        self, 
        current_weights: Dict[str, float],
        threshold: float = 0.05
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if portfolio needs rebalancing
        """
        
        # Get optimal weights for current assets
        assets = list(current_weights.keys())
        optimal = self.optimize(assets, objective="max_sharpe")
        optimal_weights = optimal['weights']
        
        # Check deviation
        needs_rebalance = False
        for asset in assets:
            current = current_weights.get(asset, 0)
            target = optimal_weights.get(asset, 0)
            
            if abs(current - target) > threshold:
                needs_rebalance = True
                break
        
        return needs_rebalance, optimal_weights if needs_rebalance else None