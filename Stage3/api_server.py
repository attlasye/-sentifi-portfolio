# Stage3/api_server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Import existing Stage3 modules
from portfolio_engine import PortfolioEngine
from crypto_pipeline import get_daily_ohlcv2
from integrated_pipeline import integrate_features
from run_ml_integrated import fit_ridge, fit_lgbm, fit_ols

# Setup
app = FastAPI(title="SentiFi Portfolio Optimizer API")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize portfolio engine
engine = PortfolioEngine()

# Request/Response Models
class OptimizationRequest(BaseModel):
    assets: List[str]
    investment_amount: float
    objective: str = "max_sharpe"  # max_sharpe, min_risk, risk_parity
    constraints: Optional[Dict] = {
        "max_weight": 0.40,
        "min_weight": 0.05
    }
    risk_tolerance: str = "moderate"  # conservative, moderate, aggressive
    use_sentiment: bool = True

class OptimizationResponse(BaseModel):
    weights: Dict[str, float]
    metrics: Dict[str, float]
    recommendation: str
    rebalance_frequency: str
    timestamp: str

class BacktestRequest(BaseModel):
    assets: List[str]
    weights: Dict[str, float]
    start_date: str
    end_date: str

# API Endpoints
@app.get("/")
async def root():
    return {"message": "SentiFi Portfolio Optimizer API", "version": "1.0"}

@app.get("/api/supported_assets")
async def get_supported_assets():
    """Return list of supported cryptocurrencies"""
    return {
        "assets": ["BTC", "ETH", "SOL", "ADA", "BNB", "XRP", "DOGE", "DOT", "AVAX", "MATIC"],
        "updated": datetime.now().isoformat()
    }

@app.post("/api/optimize", response_model=OptimizationResponse)
async def optimize_portfolio(request: OptimizationRequest):
    """
    Main optimization endpoint - generates personalized portfolio weights
    """
    try:
        logger.info(f"Optimization request for assets: {request.assets}")
        
        # Validate assets
        if len(request.assets) < 2:
            raise HTTPException(status_code=400, detail="At least 2 assets required")
        
        # Get optimization results from engine
        result = engine.optimize(
            assets=request.assets,
            objective=request.objective,
            constraints=request.constraints,
            use_sentiment=request.use_sentiment
        )
        
        # Generate recommendation based on results
        recommendation = generate_recommendation(
            result['metrics'], 
            request.risk_tolerance
        )
        
        # Determine rebalancing frequency
        rebalance_freq = "weekly" if request.risk_tolerance == "aggressive" else "monthly"
        
        return OptimizationResponse(
            weights=result['weights'],
            metrics=result['metrics'],
            recommendation=recommendation,
            rebalance_frequency=rebalance_freq,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backtest")
async def backtest_portfolio(request: BacktestRequest):
    """
    Backtest a specific portfolio configuration
    """
    try:
        results = engine.backtest(
            assets=request.assets,
            weights=request.weights,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        return {
            "performance": results['returns'],
            "metrics": results['metrics'],
            "comparison": results['benchmark_comparison']
        }
        
    except Exception as e:
        logger.error(f"Backtest error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market_sentiment")
async def get_market_sentiment():
    """
    Get current market sentiment scores
    """
    try:
        sentiment_data = engine.get_latest_sentiment()
        
        return {
            "overall": sentiment_data['overall'],
            "by_asset": sentiment_data['by_asset'],
            "trend": sentiment_data['trend'],
            "updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Sentiment fetch error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rebalance_check")
async def check_rebalance(current_weights: Dict[str, float]):
    """
    Check if portfolio needs rebalancing
    """
    try:
        needs_rebalance, suggested_weights = engine.check_rebalance_needed(
            current_weights
        )
        
        return {
            "needs_rebalance": needs_rebalance,
            "suggested_weights": suggested_weights if needs_rebalance else None,
            "reason": "Weights deviated more than 5% from target" if needs_rebalance else "Portfolio is balanced"
        }
        
    except Exception as e:
        logger.error(f"Rebalance check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper Functions
def generate_recommendation(metrics: Dict, risk_tolerance: str) -> str:
    """Generate personalized recommendation text"""
    
    sharpe = metrics.get('sharpe_ratio', 0)
    volatility = metrics.get('volatility', 0)
    
    if risk_tolerance == "conservative":
        if volatility < 0.3:
            return "This portfolio aligns well with your conservative risk profile."
        else:
            return "Consider reducing allocations to volatile assets."
    
    elif risk_tolerance == "aggressive":
        if sharpe > 2:
            return "Excellent risk-adjusted returns for aggressive growth."
        else:
            return "Consider increasing exposure to high-growth assets."
    
    else:  # moderate
        return "Balanced portfolio suitable for steady growth."

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)