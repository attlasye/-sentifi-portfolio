// src/components/OptimizerSettings.js
import React from 'react';
import { Settings, Shield, TrendingUp, Target } from 'lucide-react';
import './OptimizerSettings.css';

const OptimizerSettings = ({ 
  settings, 
  setSettings, 
  onBack, 
  onOptimize, 
  isLoading 
}) => {
  
  const handleSettingChange = (key, value) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleConstraintChange = (key, value) => {
    setSettings(prev => ({
      ...prev,
      constraints: {
        ...prev.constraints,
        [key]: parseFloat(value)
      }
    }));
  };

  return (
    <div className="optimizer-settings">
      <div className="section-header">
        <h2>Configure Optimization Strategy</h2>
        <p>Customize how your portfolio should be optimized</p>
      </div>

      {/* Optimization Objective */}
      <div className="settings-group">
        <h3>
          <Target size={20} />
          Optimization Objective
        </h3>
        <div className="objective-options">
          <label className="radio-option">
            <input
              type="radio"
              value="max_sharpe"
              checked={settings.objective === 'max_sharpe'}
              onChange={(e) => handleSettingChange('objective', e.target.value)}
            />
            <div className="option-content">
              <span className="option-title">Maximize Sharpe Ratio</span>
              <span className="option-desc">Best risk-adjusted returns</span>
            </div>
          </label>
          
          <label className="radio-option">
            <input
              type="radio"
              value="min_risk"
              checked={settings.objective === 'min_risk'}
              onChange={(e) => handleSettingChange('objective', e.target.value)}
            />
            <div className="option-content">
              <span className="option-title">Minimize Risk</span>
              <span className="option-desc">Lowest portfolio volatility</span>
            </div>
          </label>
          
          <label className="radio-option">
            <input
              type="radio"
              value="risk_parity"
              checked={settings.objective === 'risk_parity'}
              onChange={(e) => handleSettingChange('objective', e.target.value)}
            />
            <div className="option-content">
              <span className="option-title">Risk Parity</span>
              <span className="option-desc">Equal risk contribution</span>
            </div>
          </label>
        </div>
      </div>

      {/* Risk Tolerance */}
      <div className="settings-group">
        <h3>
          <Shield size={20} />
          Risk Tolerance
        </h3>
        <div className="risk-slider">
          <div className="risk-labels">
            <span className={settings.riskTolerance === 'conservative' ? 'active' : ''}>
              Conservative
            </span>
            <span className={settings.riskTolerance === 'moderate' ? 'active' : ''}>
              Moderate
            </span>
            <span className={settings.riskTolerance === 'aggressive' ? 'active' : ''}>
              Aggressive
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="2"
            value={
              settings.riskTolerance === 'conservative' ? 0 :
              settings.riskTolerance === 'moderate' ? 1 : 2
            }
            onChange={(e) => {
              const values = ['conservative', 'moderate', 'aggressive'];
              handleSettingChange('riskTolerance', values[e.target.value]);
            }}
            className="slider"
          />
        </div>
      </div>

      {/* Position Constraints */}
      <div className="settings-group">
        <h3>
          <Settings size={20} />
          Position Constraints
        </h3>
        <div className="constraints-grid">
          <div className="constraint-item">
            <label>Maximum Position Size</label>
            <div className="constraint-input">
              <input
                type="number"
                value={settings.constraints.max_weight * 100}
                onChange={(e) => handleConstraintChange('max_weight', e.target.value / 100)}
                min="10"
                max="100"
                step="5"
              />
              <span>%</span>
            </div>
          </div>
          
          <div className="constraint-item">
            <label>Minimum Position Size</label>
            <div className="constraint-input">
              <input
                type="number"
                value={settings.constraints.min_weight * 100}
                onChange={(e) => handleConstraintChange('min_weight', e.target.value / 100)}
                min="0"
                max="20"
                step="1"
              />
              <span>%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Sentiment Integration */}
      <div className="settings-group">
        <label className="checkbox-option">
          <input
            type="checkbox"
            checked={settings.useSentiment}
            onChange={(e) => handleSettingChange('useSentiment', e.target.checked)}
          />
          <span>Use sentiment analysis in optimization</span>
        </label>
      </div>

      {/* Action Buttons */}
      <div className="action-buttons">
        <button className="btn-secondary" onClick={onBack}>
          Back
        </button>
        <button 
          className="btn-primary"
          onClick={onOptimize}
          disabled={isLoading}
        >
          {isLoading ? 'Optimizing...' : 'Optimize Portfolio'}
        </button>
      </div>
    </div>
  );
};

export default OptimizerSettings;