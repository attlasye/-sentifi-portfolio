// src/components/AIAssistant.js (Updated)
import React, { useState, useEffect } from 'react';
import { MessageCircle, Send, X, Bot, User, AlertCircle } from 'lucide-react';
import './AIAssistant.css';

const AIAssistant = ({ 
  optimizationResults, 
  selectedAssets, 
  settings, 
  currentStep,
  apiKey,
  isDemoMode 
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [hasApiKey, setHasApiKey] = useState(false);

  useEffect(() => {
    // Check if we have an API key
    setHasApiKey(!!apiKey);
    
    // Set initial message based on API key availability
    if (isDemoMode) {
      setMessages([{
        role: 'assistant',
        content: "Hi! I'm your SentiFi AI Assistant. I'm running in demo mode with limited capabilities. To unlock full AI features, please configure your Gemini API key."
      }]);
    } else if (!apiKey) {
      setMessages([{
        role: 'assistant',
        content: "Hi! I'm your SentiFi AI Assistant. To enable AI-powered insights, please add your Google Gemini API key in the configuration. For now, I can provide basic information about the strategies."
      }]);
    } else {
      setMessages([{
        role: 'assistant',
        content: "Hi! I'm your SentiFi AI Assistant. I can help you understand investment strategies, analyze your portfolio performance, and provide insights about our optimization engine. What would you like to know?"
      }]);
    }
  }, [apiKey, isDemoMode]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = inputMessage.trim();
    setInputMessage('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      if (!apiKey) {
        // Provide basic responses without API
        const basicResponse = getBasicResponse(userMessage);
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: basicResponse 
        }]);
      } else {
        // Use Gemini API
        const response = await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key=${apiKey}`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              contents: [{
                parts: [{ 
                  text: `${createContext()}\n\nUser question: ${userMessage}\n\nProvide a helpful response.` 
                }]
              }],
              generationConfig: {
                temperature: 0.7,
                topK: 40,
                topP: 0.95,
                maxOutputTokens: 1024,
              }
            })
          }
        );

        if (!response.ok) {
          throw new Error('API request failed');
        }

        const data = await response.json();
        const responseText = data.candidates?.[0]?.content?.parts?.[0]?.text || 
                           'I apologize, but I couldn\'t generate a response.';
        
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: responseText 
        }]);
      }
    } catch (error) {
      console.error('AI Assistant Error:', error);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'I apologize, but I encountered an error. Please try again or check your API key configuration.' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const getBasicResponse = (question) => {
    const lowerQuestion = question.toLowerCase();
    
    if (lowerQuestion.includes('mv') || lowerQuestion.includes('mean variance')) {
      return 'Mean-Variance Optimization (MVO) seeks the optimal balance between expected returns and portfolio risk. It uses your ML predictions to maximize returns while controlling volatility.';
    }
    
    if (lowerQuestion.includes('risk parity')) {
      return 'Risk Parity ensures each asset contributes equally to portfolio risk. Our hybrid version blends this with ML predictions for enhanced performance.';
    }
    
    if (lowerQuestion.includes('sentiment')) {
      return 'Our sentiment analysis uses VADER enhanced with crypto-specific terms. We found that price drives sentiment, not vice versa, so we use it for risk management.';
    }
    
    if (lowerQuestion.includes('rebalance')) {
      return 'We recommend weekly rebalancing for aggressive strategies and monthly for conservative ones. Rebalancing maintains your target allocations as prices change.';
    }
    
    return 'I can provide basic information about our strategies. For detailed AI-powered insights, please configure your Gemini API key. What specific aspect would you like to know about?';
  };

  const createContext = () => {
    const context = `
You are an AI assistant for SentiFi Portfolio Optimizer.

Current Status:
- Step: ${currentStep === 1 ? 'Asset Selection' : currentStep === 2 ? 'Strategy Configuration' : 'Results'}
- Selected Assets: ${selectedAssets.join(', ') || 'None'}
- Optimization Settings: ${settings?.objective || 'Not configured'}
${optimizationResults ? `
- Portfolio Results:
  - Expected Return: ${optimizationResults.metrics?.expected_return?.toFixed(1)}%
  - Volatility: ${optimizationResults.metrics?.volatility?.toFixed(1)}%
  - Sharpe Ratio: ${optimizationResults.metrics?.sharpe_ratio?.toFixed(2)}
` : ''}

Help the user understand the optimization process and results.`;
    return context;
  };

  const quickActions = hasApiKey ? [
    "Explain my optimization results",
    "What is Risk Parity?",
    "How does sentiment affect portfolios?",
    "Should I rebalance now?"
  ] : [
    "What is Mean-Variance Optimization?",
    "Explain Risk Parity",
    "How does rebalancing work?",
    "What are the strategy options?"
  ];

  const handleQuickAction = (action) => {
    setInputMessage(action);
    setTimeout(() => {
      handleSendMessage();
    }, 100);
  };

  return (
    <>
      <button 
        className={`ai-assistant-button ${isOpen ? 'hidden' : ''}`}
        onClick={() => setIsOpen(true)}
        type="button"
      >
        <Bot size={24} />
        <span>AI Assistant</span>
      </button>

      {isOpen && (
        <div className="ai-assistant-container">
          <div className="ai-assistant-header">
            <div className="header-title">
              <Bot size={20} />
              <h3>SentiFi AI Assistant</h3>
              {!hasApiKey && <span className="no-api-badge">Limited Mode</span>}
            </div>
            <button 
              onClick={() => setIsOpen(false)} 
              className="close-button"
              type="button"
            >
              <X size={20} />
            </button>
          </div>

          {!hasApiKey && (
            <div className="api-warning">
              <AlertCircle size={16} />
              <span>Add Gemini API key for full AI features</span>
            </div>
          )}

          <div className="ai-assistant-messages">
            {messages.map((message, index) => (
              <div key={index} className={`message ${message.role}`}>
                <div className="message-icon">
                  {message.role === 'assistant' ? <Bot size={16} /> : <User size={16} />}
                </div>
                <div className="message-content">
                  {message.content}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="message assistant">
                <div className="message-icon">
                  <Bot size={16} />
                </div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="quick-actions">
            {quickActions.map((action, index) => (
              <button
                key={index}
                className="quick-action-button"
                onClick={() => handleQuickAction(action)}
                disabled={isLoading}
                type="button"
              >
                {action}
              </button>
            ))}
          </div>

          <div className="ai-assistant-input">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder={hasApiKey ? "Ask me anything..." : "Ask about strategies..."}
              disabled={isLoading}
            />
            <button 
              onClick={handleSendMessage} 
              disabled={isLoading || !inputMessage.trim()}
              className="send-button"
              type="button"
            >
              <Send size={18} />
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default AIAssistant;