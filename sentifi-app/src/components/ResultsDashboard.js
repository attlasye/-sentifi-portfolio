// src/components/ResultsDashboard.js
import React, { useState } from 'react';
import {
  PieChart, Pie, Cell, BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { 
  TrendingUp, DollarSign, Shield, AlertTriangle, 
  RefreshCw, Plus, Download, Share2, Info,
  ChevronDown, ChevronUp, Target, Activity
} from 'lucide-react';
import './ResultsDashboard.css';

const ResultsDashboard = ({ 
  results, 
  investmentAmount, 
  selectedAssets,
  onNewOptimization,
  onRebalanceCheck,
  isDemoMode 
}) => {
  const [expandedSection, setExpandedSection] = useState('overview');
  const [showDetails, setShowDetails] = useState(false);

  if (!results) {
    return (
      <div className="dashboard-container">
        <div className="no-results">
          <p>No optimization results yet.</p>
        </div>
      </div>
    );
  }

  // Prepare data for charts
  const pieData = Object.entries(results.weights).map(([asset, weight]) => ({
    name: asset,
    value: Math.round(weight * 100),
    amount: Math.round(weight * investmentAmount)
  }));

  const COLORS = ['#8B5CF6', '#7C3AED', '#A78BFA', '#6D28D9', '#9333EA', '#C084FC'];

  // Mock historical data for performance chart
  const performanceData = [
    { month: 'Jan', portfolio: 100, benchmark: 100 },
    { month: 'Feb', portfolio: 108, benchmark: 103 },
    { month: 'Mar', portfolio: 115, benchmark: 105 },
    { month: 'Apr', portfolio: 112, benchmark: 102 },
    { month: 'May', portfolio: 125, benchmark: 108 },
    { month: 'Jun', portfolio: 135, benchmark: 110 },
  ];

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatPercent = (value) => {
    return `${value.toFixed(2)}%`;
  };

  const getRiskLevel = (volatility) => {
    if (volatility < 30) return { level: 'Low', color: 'green' };
    if (volatility < 50) return { level: 'Medium', color: 'yellow' };
    return { level: 'High', color: 'red' };
  };

  const riskInfo = getRiskLevel(results.metrics.volatility);

  const handleExport = () => {
    const exportData = {
      portfolio: results,
      investment: investmentAmount,
      assets: selectedAssets,
      timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `portfolio-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="dashboard-container">
      {/* Header Section */}
      <div className="dashboard-header">
        <div className="header-main">
          <h2>Optimization Results</h2>
          {isDemoMode && <span className="demo-badge">Demo Mode</span>}
        </div>
        <div className="header-actions">
          <button className="icon-btn" onClick={handleExport} title="Export">
            <Download size={18} />
          </button>
          <button className="icon-btn" title="Share">
            <Share2 size={18} />
          </button>
        </div>
      </div>

      {/* Key Metrics Cards */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-icon returns">
            <TrendingUp size={20} />
          </div>
          <div className="metric-content">
            <span className="metric-label">Expected Annual Return</span>
            <span className="metric-value">{formatPercent(results.metrics.expected_return)}</span>
            <span className="metric-change positive">+{formatPercent(results.metrics.expected_return - 20)}</span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon risk">
            <Shield size={20} />
          </div>
          <div className="metric-content">
            <span className="metric-label">Portfolio Risk</span>
            <span className="metric-value">{formatPercent(results.metrics.volatility)}</span>
            <span className={`risk-level ${riskInfo.color}`}>{riskInfo.level} Risk</span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon sharpe">
            <Target size={20} />
          </div>
          <div className="metric-content">
            <span className="metric-label">Sharpe Ratio</span>
            <span className="metric-value">{results.metrics.sharpe_ratio.toFixed(2)}</span>
            <span className="metric-info">Risk-Adjusted Returns</span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon diversification">
            <Activity size={20} />
          </div>
          <div className="metric-content">
            <span className="metric-label">Diversification</span>
            <span className="metric-value">{results.metrics.num_assets} Assets</span>
            <span className="metric-info">Max: {formatPercent(results.metrics.max_weight * 100)}</span>
          </div>
        </div>
      </div>

      {/* Portfolio Allocation */}
      <div className="chart-section">
        <div className="section-header">
          <h3>Portfolio Allocation</h3>
          <button 
            className="toggle-btn"
            onClick={() => setShowDetails(!showDetails)}
          >
            {showDetails ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
            Details
          </button>
        </div>

        <div className="allocation-content">
          <div className="pie-chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => `${name}: ${value}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => `${value}%`} />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="allocation-table">
            <h4>Investment Breakdown</h4>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Asset</th>
                    <th>Weight</th>
                    <th>Amount</th>
                  </tr>
                </thead>
                <tbody>
                  {pieData.map((item, index) => (
                    <tr key={index}>
                      <td>
                        <div className="asset-cell">
                          <span 
                            className="asset-color" 
                            style={{ backgroundColor: COLORS[index % COLORS.length] }}
                          />
                          {item.name}
                        </div>
                      </td>
                      <td>{item.value}%</td>
                      <td>{formatCurrency(item.amount)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Projection */}
      <div className="chart-section">
        <div className="section-header">
          <h3>Performance Projection</h3>
        </div>
        <div className="performance-chart">
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="month" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'var(--bg-card)', 
                  border: '1px solid var(--border-color)' 
                }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="portfolio" 
                stroke="#8B5CF6" 
                strokeWidth={2}
                name="Your Portfolio"
              />
              <Line 
                type="monotone" 
                dataKey="benchmark" 
                stroke="#6B7280" 
                strokeWidth={2}
                name="Market Benchmark"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recommendations */}
      <div className="recommendations-section">
        <h3>Recommendations</h3>
        <div className="recommendation-cards">
          <div className="recommendation-card">
            <AlertTriangle size={20} className="warning-icon" />
            <div className="recommendation-content">
              <h4>Rebalancing Schedule</h4>
              <p>{results.recommendation || "Rebalance weekly for optimal performance"}</p>
            </div>
          </div>
          
          <div className="recommendation-card">
            <Info size={20} className="info-icon" />
            <div className="recommendation-content">
              <h4>Risk Management</h4>
              <p>Consider setting stop-loss orders at -10% for each position</p>
            </div>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="actions-section">
        <button 
          className="action-button secondary"
          onClick={() => onRebalanceCheck(results.weights)}
        >
          <RefreshCw size={18} />
          <span>Check Rebalance Need</span>
        </button>
        
        <button 
          className="action-button primary"
          onClick={onNewOptimization}
        >
          <Plus size={18} />
          <span>New Optimization</span>
        </button>
      </div>
    </div>
  );
};

export default ResultsDashboard;