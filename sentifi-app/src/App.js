// src/App.js (Complete Version with Config Page)
import React, { useState, useEffect } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { 
  TrendingUp, Activity, Brain, Settings,
  DollarSign, AlertCircle, RefreshCw, Info, LogOut
} from 'lucide-react';

// Components
import ConfigPage from './components/ConfigPage';
import PortfolioBuilder from './components/PortfolioBuilder';
import OptimizerSettings from './components/OptimizerSettings';
import ResultsDashboard from './components/ResultsDashboard';
import AIAssistant from './components/AIAssistant';

// Services
import apiService from './services/apiService';

import './App.css';

function App() {
  // Configuration State
  const [isConfigured, setIsConfigured] = useState(false);
  const [apiKeys, setApiKeys] = useState(null);
  
  // Application State
  const [activeStep, setActiveStep] = useState(1);
  const [selectedAssets, setSelectedAssets] = useState([]);
  const [investmentAmount, setInvestmentAmount] = useState(10000);
  const [optimizerSettings, setOptimizerSettings] = useState({
    objective: 'max_sharpe',
    riskTolerance: 'moderate',
    constraints: {
      max_weight: 0.40,
      min_weight: 0.05
    },
    useSentiment: true
  });
  const [optimizationResults, setOptimizationResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [marketSentiment, setMarketSentiment] = useState(null);
  const [supportedAssets, setSupportedAssets] = useState([]);
  const [isDemoMode, setIsDemoMode] = useState(false);

  // Check for existing configuration on mount
 useEffect(() => {
  const savedConfig = localStorage.getItem('sentifi_config');
  if (savedConfig) {
    try {
      const config = JSON.parse(savedConfig);
      setApiKeys(config);
      setIsConfigured(true);
      // 不再需要设置CryptoCompare API key
    } catch (error) {
      console.error('Invalid saved configuration');
      localStorage.removeItem('sentifi_config');
    }
  }
}, []);

const handleConfigComplete = (config) => {
  setApiKeys(config);
  setIsConfigured(true);
  toast.success('Welcome to SentiFi! 🚀');
};

  const handleLogout = () => {
    localStorage.removeItem('sentifi_config');
    setIsConfigured(false);
    setApiKeys(null);
    setActiveStep(1);
    setSelectedAssets([]);
    setOptimizationResults(null);
    toast.info('Logged out successfully');
  };

  const loadInitialData = async () => {
    try {
      // Load market sentiment
      const sentiment = await apiService.getMarketSentiment();
      setMarketSentiment(sentiment);

      // Load supported assets
      const assets = await apiService.getSupportedAssets();
      setSupportedAssets(assets.assets);
    } catch (error) {
      console.error('Error loading initial data:', error);
      if (error.response?.status === 401) {
        toast.error('Invalid API key. Please reconfigure.');
        handleLogout();
      } else {
        toast.error('Failed to load market data. Please check your connection.');
      }
    }
  };

  const loadDemoData = () => {
    // Set demo data
    setMarketSentiment({
      overall: 65,
      trend: 'bullish',
      by_asset: {
        BTC: 70,
        ETH: 62,
        SOL: 58
      }
    });
    
    setSupportedAssets(['BTC', 'ETH', 'SOL', 'ADA', 'MATIC']);
    
    toast.info('Running in demo mode with limited features');
  };

  // Handle optimization
  const handleOptimize = async () => {
    if (selectedAssets.length < 2) {
      toast.error('Please select at least 2 assets');
      return;
    }

    if (investmentAmount < 100) {
      toast.error('Minimum investment amount is $100');
      return;
    }

    if (isDemoMode) {
      // Generate demo results
      const demoResults = generateDemoResults(selectedAssets, investmentAmount);
      setOptimizationResults(demoResults);
      setActiveStep(3);
      toast.success('Demo optimization complete!');
      return;
    }

    setIsLoading(true);
    try {
      const result = await apiService.optimizePortfolio({
        assets: selectedAssets,
        investment_amount: investmentAmount,
        objective: optimizerSettings.objective,
        constraints: optimizerSettings.constraints,
        risk_tolerance: optimizerSettings.riskTolerance,
        use_sentiment: optimizerSettings.useSentiment
      });
      
      setOptimizationResults(result);
      setActiveStep(3);
      toast.success('Portfolio optimized successfully!');
    } catch (error) {
      console.error('Optimization error:', error);
      toast.error('Optimization failed: ' + (error.response?.data?.detail || error.message));
    } finally {
      setIsLoading(false);
    }
  };

  // Generate demo results for demo mode
  const generateDemoResults = (assets, amount) => {
    const weights = {};
    const remaining = 1.0;
    const weightPerAsset = remaining / assets.length;
    
    assets.forEach(asset => {
      weights[asset] = weightPerAsset + (Math.random() - 0.5) * 0.1;
    });
    
    // Normalize weights
    const sum = Object.values(weights).reduce((a, b) => a + b, 0);
    Object.keys(weights).forEach(key => {
      weights[key] = weights[key] / sum;
    });
    
    return {
      weights,
      metrics: {
        expected_return: 45 + Math.random() * 30,
        volatility: 35 + Math.random() * 20,
        sharpe_ratio: 1.2 + Math.random() * 0.8,
        max_weight: Math.max(...Object.values(weights)),
        min_weight: Math.min(...Object.values(weights)),
        num_assets: assets.length
      },
      recommendation: "Demo mode: This is a simulated optimization result.",
      rebalance_frequency: "weekly",
      timestamp: new Date().toISOString()
    };
  };

  // Handle rebalance check
  const handleRebalanceCheck = async (currentWeights) => {
    if (isDemoMode) {
      toast.info('Rebalance check not available in demo mode');
      return null;
    }

    try {
      const result = await apiService.checkRebalance(currentWeights);
      if (result.needs_rebalance) {
        toast.warning('Your portfolio needs rebalancing: ' + result.reason);
        return result.suggested_weights;
      } else {
        toast.info('Your portfolio is well balanced');
        return null;
      }
    } catch (error) {
      console.error('Rebalance check error:', error);
      toast.error('Failed to check rebalance status');
      return null;
    }
  };

  // Reset optimization
  const handleNewOptimization = () => {
    setActiveStep(1);
    setOptimizationResults(null);
    setSelectedAssets([]);
    setInvestmentAmount(10000);
    setOptimizerSettings({
      objective: 'max_sharpe',
      riskTolerance: 'moderate',
      constraints: {
        max_weight: 0.40,
        min_weight: 0.05
      },
      useSentiment: true
    });
  };

  // Render content based on active step
  const renderStep = () => {
    switch(activeStep) {
      case 1:
        return (
          <PortfolioBuilder
            selectedAssets={selectedAssets}
            setSelectedAssets={setSelectedAssets}
            investmentAmount={investmentAmount}
            setInvestmentAmount={setInvestmentAmount}
            onNext={() => setActiveStep(2)}
            marketSentiment={marketSentiment}
            supportedAssets={supportedAssets}
            isDemoMode={isDemoMode}
          />
        );
      
      case 2:
        return (
          <OptimizerSettings
            settings={optimizerSettings}
            setSettings={setOptimizerSettings}
            selectedAssets={selectedAssets}
            onBack={() => setActiveStep(1)}
            onOptimize={handleOptimize}
            isLoading={isLoading}
          />
        );
      
      case 3:
        return (
          <ResultsDashboard
            results={optimizationResults}
            investmentAmount={investmentAmount}
            selectedAssets={selectedAssets}
            onNewOptimization={handleNewOptimization}
            onRebalanceCheck={handleRebalanceCheck}
            isDemoMode={isDemoMode}
          />
        );
      
      default:
        return null;
    }
  };

  // Show configuration page if not configured
  if (!isConfigured) {
    return <ConfigPage onConfigComplete={handleConfigComplete} />;
  }

  // Main application
  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-icon">
              <TrendingUp size={28} />
            </div>
            <div className="logo-text">
              <h1>SentiFi Portfolio Optimizer</h1>
              <p className="tagline">
                {isDemoMode ? 'Demo Mode' : 'AI-Powered Crypto Portfolio Optimization'}
              </p>
            </div>
          </div>
          
          <div className="header-stats">
            {marketSentiment && (
              <>
                <div className="stat-item">
                  <span className="stat-label">Market Sentiment</span>
                  <span className={`stat-value ${
                    marketSentiment.overall > 60 ? 'positive' : 
                    marketSentiment.overall < 40 ? 'negative' : 'neutral'
                  }`}>
                    {marketSentiment.overall.toFixed(0)}
                  </span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Trend</span>
                  <span className={`stat-value ${marketSentiment.trend}`}>
                    {marketSentiment.trend.charAt(0).toUpperCase() + marketSentiment.trend.slice(1)}
                  </span>
                </div>
              </>
            )}
            <button className="logout-btn" onClick={handleLogout} title="Logout">
              <LogOut size={18} />
            </button>
          </div>
        </div>
      </header>

      {/* Progress Steps */}
      <div className="progress-container">
        <div className="progress-steps">
          <div className={`step ${activeStep >= 1 ? 'active' : ''} ${activeStep === 1 ? 'current' : ''}`}>
            <div className="step-number">1</div>
            <div className="step-info">
              <span className="step-title">Select Assets</span>
              <span className="step-desc">Choose cryptocurrencies</span>
            </div>
          </div>
          
          <div className="step-connector"></div>
          
          <div className={`step ${activeStep >= 2 ? 'active' : ''} ${activeStep === 2 ? 'current' : ''}`}>
            <div className="step-number">2</div>
            <div className="step-info">
              <span className="step-title">Configure Strategy</span>
              <span className="step-desc">Set your preferences</span>
            </div>
          </div>
          
          <div className="step-connector"></div>
          
          <div className={`step ${activeStep >= 3 ? 'active' : ''} ${activeStep === 3 ? 'current' : ''}`}>
            <div className="step-number">3</div>
            <div className="step-info">
              <span className="step-title">View Results</span>
              <span className="step-desc">Get optimized portfolio</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="main-container">
        <div className="content-wrapper">
          {renderStep()}
        </div>
      </main>

      {/* Info Banner */}
      {activeStep === 1 && (
        <div className="info-banner">
          <Info size={20} />
          <p>
            {isDemoMode 
              ? "You're in demo mode. Results are simulated. Configure API keys for real data."
              : "Our AI-powered optimizer uses machine learning and sentiment analysis to create personalized portfolio recommendations."}
          </p>
        </div>
      )}

      {/* AI Assistant */}
      <AIAssistant 
        optimizationResults={optimizationResults}
        selectedAssets={selectedAssets}
        settings={optimizerSettings}
        currentStep={activeStep}
        apiKey={apiKeys?.geminiApiKey}
        isDemoMode={isDemoMode}
      />

      {/* Toast Notifications */}
      <ToastContainer 
        position="bottom-right"
        theme="dark"
        autoClose={4000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
      />
    </div>
  );
}

export default App;