// src/services/apiService.js
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://sentifi-portfolio-attlasye.replit.app';

class ApiService {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });
    this.apiKey = null;
  }

  setApiKey(key) {
    this.apiKey = key;
    // Add API key to all requests
    this.client.defaults.headers.common['X-API-Key'] = key;
  }

  async getSupportedAssets() {
    const response = await this.client.get('/api/supported_assets');
    return response.data;
  }

  async optimizePortfolio(params) {
    const response = await this.client.post('/api/optimize', params);
    return response.data;
  }

  async backtest(params) {
    const response = await this.client.post('/api/backtest', params);
    return response.data;
  }

  async getMarketSentiment() {
    const response = await this.client.get('/api/market_sentiment');
    return response.data;
  }

  async checkRebalance(currentWeights) {
    const response = await this.client.post('/api/rebalance_check', currentWeights);
    return response.data;
  }
}

export default new ApiService();
