// src/components/ConfigPage.js
import React, { useState } from 'react';
import { Bot, AlertCircle, CheckCircle, ArrowRight, Book, Github, Sparkles } from 'lucide-react';
import './ConfigPage.css';

const ConfigPage = ({ onConfigComplete }) => {
  const [geminiApiKey, setGeminiApiKey] = useState('');
  const [tested, setTested] = useState(false);
  const [testResult, setTestResult] = useState(null);

  const handleTestGeminiKey = async () => {
    if (!geminiApiKey) return;
    
    setTested(true);
    try {
      // Test Gemini API
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models?key=${geminiApiKey}`
      );
      
      if (response.ok) {
        setTestResult('success');
      } else {
        setTestResult('failed');
      }
    } catch (error) {
      setTestResult('failed');
    }
  };

  const handleContinueWithAI = () => {
    localStorage.setItem('sentifi_config', JSON.stringify({ 
      geminiApiKey: geminiApiKey 
    }));
    onConfigComplete({ geminiApiKey });
  };

  const handleContinueWithoutAI = () => {
    localStorage.setItem('sentifi_config', JSON.stringify({ 
      geminiApiKey: '' 
    }));
    onConfigComplete({ geminiApiKey: '' });
  };

  return (
    <div className="config-page">
      <div className="config-container">
        <div className="config-header">
          <div className="logo-section">
            <div className="logo-icon">
              <Sparkles size={32} />
            </div>
            <div>
              <h1>Welcome to SentiFi Portfolio Optimizer</h1>
              <p>AI-Powered Cryptocurrency Portfolio Management</p>
            </div>
          </div>
        </div>

        <div className="config-content">
          <div className="welcome-section">
            <h2>Get Started</h2>
            <p>
              SentiFi uses advanced machine learning and sentiment analysis to optimize 
              your cryptocurrency portfolio. Our system analyzes market data and news 
              sentiment to provide personalized investment recommendations.
            </p>
          </div>

          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">📊</div>
              <h3>ML-Driven Optimization</h3>
              <p>Advanced algorithms analyze market patterns</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">💭</div>
              <h3>Sentiment Analysis</h3>
              <p>Real-time news sentiment tracking</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">⚡</div>
              <h3>Real-time Updates</h3>
              <p>Live market data and rebalancing alerts</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🎯</div>
              <h3>Risk Management</h3>
              <p>Multiple strategies for different risk levels</p>
            </div>
          </div>

          <div className="ai-section">
            <div className="ai-header">
              <Bot size={20} />
              <h3>Optional: Enable AI Assistant</h3>
            </div>
            <p className="ai-description">
              Add your Google Gemini API key to unlock AI-powered insights and 
              personalized explanations. 
              <a href="https://makersuite.google.com/app/apikey" target="_blank" rel="noopener noreferrer">
                {' '}Get free API key →
              </a>
            </p>
            
            <div className="input-group">
              <input
                type="password"
                placeholder="Enter Gemini API key (optional)"
                value={geminiApiKey}
                onChange={(e) => {
                  setGeminiApiKey(e.target.value);
                  setTested(false);
                  setTestResult(null);
                }}
                className="api-input"
              />
              {testResult === 'success' && (
                <CheckCircle className="status-icon success" size={20} />
              )}
              {testResult === 'failed' && (
                <AlertCircle className="status-icon error" size={20} />
              )}
            </div>
            
            {geminiApiKey && !tested && (
              <button 
                className="test-ai-btn"
                onClick={handleTestGeminiKey}
              >
                Test AI Connection
              </button>
            )}
            
            {testResult === 'failed' && (
              <p className="error-message">
                Invalid API key. Please check and try again, or continue without AI.
              </p>
            )}
          </div>

          <div className="action-section">
            {geminiApiKey && testResult === 'success' ? (
              <button 
                className="primary-btn"
                onClick={handleContinueWithAI}
              >
                <Sparkles size={18} />
                Continue with AI Assistant
              </button>
            ) : (
              <button 
                className="primary-btn"
                onClick={handleContinueWithoutAI}
              >
                <ArrowRight size={18} />
                Start Optimizing
              </button>
            )}
            
            {geminiApiKey && testResult !== 'success' && (
              <button 
                className="secondary-btn"
                onClick={handleContinueWithoutAI}
              >
                Skip AI Setup
              </button>
            )}
          </div>
        </div>

        <div className="config-footer">
          <p className="footer-text">
            Powered by advanced ML models trained on cryptocurrency market data
          </p>
        </div>
      </div>
    </div>
  );
};

export default ConfigPage;