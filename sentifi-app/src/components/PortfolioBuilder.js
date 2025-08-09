// src/components/PortfolioBuilder.js
import React, { useState, useEffect } from 'react';
import Select from 'react-select';
import { Search, Plus, DollarSign, TrendingUp, AlertCircle } from 'lucide-react';
import apiService from '../services/apiService';
import './PortfolioBuilder.css';

const PortfolioBuilder = ({ 
  selectedAssets, 
  setSelectedAssets, 
  investmentAmount, 
  setInvestmentAmount,
  onNext,
  marketSentiment 
}) => {
  const [availableAssets, setAvailableAssets] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    loadAvailableAssets();
  }, []);

  const loadAvailableAssets = async () => {
    try {
      const data = await apiService.getSupportedAssets();
      setAvailableAssets(data.assets.map(asset => ({
        value: asset,
        label: asset,
        sentiment: marketSentiment?.by_asset?.[asset] || 50
      })));
    } catch (error) {
      console.error('Error loading assets:', error);
    }
  };

  const handleAssetToggle = (asset) => {
    if (selectedAssets.includes(asset.value)) {
      setSelectedAssets(selectedAssets.filter(a => a !== asset.value));
    } else {
      setSelectedAssets([...selectedAssets, asset.value]);
    }
  };

  const getSentimentColor = (score) => {
    if (score > 60) return '#10B981';
    if (score < 40) return '#EF4444';
    return '#F59E0B';
  };

  return (
    <div className="portfolio-builder">
      <div className="section-header">
        <h2>Build Your Portfolio</h2>
        <p>Select cryptocurrencies and set your investment amount</p>
      </div>

      {/* Investment Amount */}
      <div className="investment-section">
        <label className="input-label">
          <DollarSign size={20} />
          Investment Amount (USD)
        </label>
        <input
          type="number"
          className="amount-input"
          value={investmentAmount}
          onChange={(e) => setInvestmentAmount(parseFloat(e.target.value) || 0)}
          min="100"
          step="100"
        />
      </div>

      {/* Asset Selection */}
      <div className="assets-section">
        <label className="input-label">
          <Search size={20} />
          Select Cryptocurrencies
        </label>
        
        <div className="assets-grid">
          {availableAssets.map(asset => (
            <div
              key={asset.value}
              className={`asset-card ${selectedAssets.includes(asset.value) ? 'selected' : ''}`}
              onClick={() => handleAssetToggle(asset)}
            >
              <div className="asset-header">
                <span className="asset-symbol">{asset.label}</span>
                <div 
                  className="sentiment-indicator"
                  style={{ backgroundColor: getSentimentColor(asset.sentiment) }}
                >
                  {asset.sentiment.toFixed(0)}
                </div>
              </div>
              <div className="asset-footer">
                <span className="sentiment-label">Sentiment Score</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Selected Assets Summary */}
      {selectedAssets.length > 0 && (
        <div className="selection-summary">
          <h3>Selected Assets ({selectedAssets.length})</h3>
          <div className="selected-tags">
            {selectedAssets.map(asset => (
              <span key={asset} className="asset-tag">
                {asset}
                <button 
                  onClick={() => setSelectedAssets(selectedAssets.filter(a => a !== asset))}
                  className="remove-btn"
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="action-buttons">
        <button 
          className="btn-primary"
          onClick={onNext}
          disabled={selectedAssets.length < 2}
        >
          Continue to Strategy Settings
        </button>
      </div>

      {selectedAssets.length < 2 && (
        <div className="warning-message">
          <AlertCircle size={16} />
          Please select at least 2 assets to continue
        </div>
      )}
    </div>
  );
};

export default PortfolioBuilder;