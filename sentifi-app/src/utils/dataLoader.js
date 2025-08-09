// src/utils/dataLoader.js
import Papa from 'papaparse';

class DataLoader {
  constructor() {
    this.cache = {};
  }

  async loadCSV(filePath) {
    if (this.cache[filePath]) {
      return this.cache[filePath];
    }

    try {
      const response = await fetch(filePath);
      const text = await response.text();
      const result = Papa.parse(text, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true
      });
      
      this.cache[filePath] = result.data;
      return result.data;
    } catch (error) {
      console.error(`Error loading ${filePath}:`, error);
      return null;
    }
  }

  async loadTextFile(filePath) {
    try {
      const response = await fetch(filePath);
      const text = await response.text();
      return text;
    } catch (error) {
      console.error(`Error loading ${filePath}:`, error);
      return null;
    }
  }

  async loadAllData() {
    // Define all data file paths
    const dataPaths = {
      portfolioReturns: '/data/integrated_portfolios.csv',
      cumulativeReturns: '/data/cumulative_returns_data.csv',
      cumulativeLogReturns: '/data/cumulative_log_returns_data.csv',
      monthlyReturns: '/data/monthly_returns_data.csv',
      correlationMatrix: '/data/correlation_matrix_data.csv',
      drawdownTimeline: '/data/drawdown_timeline_data.csv',
      mlPredictions: '/data/ml_predictions_integrated.csv',
      integratedFeatures: '/data/integrated_features.csv',
      sentimentTimeseries: '/data/compound_timeseries_daily.csv',
      metricsTable: '/data/metrics_table_sorted.csv'
    };

    const loadedData = {};
  
    // Load all CSV files
    const loadPromises = Object.entries(dataPaths).map(async ([key, path]) => {
      const data = await this.loadCSV(path);
      return { key, data };
    });
  
    // Load text files separately
    const validationMetrics = await this.loadTextFile('/data/statistical_validation_metrics.txt');
    if (validationMetrics) {
      loadedData.validationMetrics = validationMetrics;
    }
  
    const results = await Promise.all(loadPromises);
  
    results.forEach(({ key, data }) => {
      if (data) {
        loadedData[key] = data;
      }
    });
  
    return this.processLoadedData(loadedData);
  }

  processLoadedData(rawData) {
    // Process and structure the data for the app
    const processedData = {
      strategies: {},
      historicalPerformance: [],
      monthlyReturns: {},
      mlModels: {},
      sentimentAnalysis: {},
      featureImportance: [],
      correlationMatrix: {},
      sentimentTimeline: []
    };

    // Process performance metrics
    if (rawData.metricsTable) {
      rawData.metricsTable.forEach(row => {
        // Handle the case where first column might not have a header
        let strategyKey;
        if (row.hasOwnProperty('')) {
          strategyKey = row[''];
        } else {
          strategyKey = Object.keys(row)[0];
        }
      
        processedData.strategies[strategyKey] = {
          name: this.formatStrategyName(strategyKey),
          annualReturn: parseFloat(row['Ave.Return(%)']) || 0,
          volatility: parseFloat(row['Volatility(%)']) || 0,
          sharpeRatio: parseFloat(row['Sharpe']) || 0,
          maxDrawdown: parseFloat(row['Max_Drawdown(%)']) || 0,
          netSharpe: parseFloat(row['Net_Sharpe']) || 0,
          turnover: parseFloat(row['Turnover(Annual)']) || 0
        };
      });
    }

    // Process cumulative returns for historical performance
    if (rawData.cumulativeReturns) {
      processedData.historicalPerformance = rawData.cumulativeReturns.map(row => {
        const dataPoint = { month: row.date };
        Object.keys(row).forEach(key => {
          if (key !== 'date') {
            dataPoint[key] = row[key] * 1000000; // Convert to dollar values
          }
        });
        return dataPoint;
      });
    }

    // Process monthly returns for heatmap
    if (rawData.monthlyReturns) {
      processedData.monthlyReturns = {};
      
      rawData.monthlyReturns.forEach(row => {
        const month = row.month;
        Object.keys(row).forEach(strategy => {
          if (strategy !== 'month') {
            if (!processedData.monthlyReturns[strategy]) {
              processedData.monthlyReturns[strategy] = [];
            }
            processedData.monthlyReturns[strategy].push({
              month: month,
              value: row[strategy]
            });
          }
        });
      });
      
      // Convert to array format for display
      Object.keys(processedData.monthlyReturns).forEach(strategy => {
        processedData.monthlyReturns[strategy] = processedData.monthlyReturns[strategy].map(d => d.value);
      });
    }
    
    // Process validation metrics from text file
    if (rawData.validationMetrics) {
      processedData.mlModels = this.parseValidationMetrics(rawData.validationMetrics);
    } else {
      // Fallback to hardcoded values if file not found
      processedData.mlModels = {
        ridge: { r2: 0.0008, directionAccuracy: 48.97, name: 'Ridge Regression' },
        lightgbm: { r2: -0.0211, directionAccuracy: 46.61, name: 'LightGBM' },
        ols: { r2: 0.0007, directionAccuracy: 48.95, name: 'OLS' },
        ensemble: { r2: -0.0028, directionAccuracy: 48.72, name: 'Ensemble Model' }
      };
    }

    // Process sentiment data
    if (rawData.sentimentTimeseries) {
      const latestSentiment = rawData.sentimentTimeseries[rawData.sentimentTimeseries.length - 1];
      processedData.sentimentAnalysis = {
        currentScore: latestSentiment.Overall || 50,
        weeklyAverage: this.calculateAverage(rawData.sentimentTimeseries.slice(-7), 'Overall'),
        monthlyAverage: this.calculateAverage(rawData.sentimentTimeseries.slice(-30), 'Overall'),
        topSentiments: {
          BTC: latestSentiment.BTC || 50,
          ETH: latestSentiment.ETH || 50,
          SOL: latestSentiment.SOL || 50,
          ADA: latestSentiment.ADA || 50
        }
      };

      // Prepare timeline data for chart
      processedData.sentimentTimeline = rawData.sentimentTimeseries.map(row => ({
        date: row.date,
        Overall: parseFloat(row.Overall) || 50,
        BTC: parseFloat(row.BTC) || 50,
        ETH: parseFloat(row.ETH) || 50,
        SOL: parseFloat(row.SOL) || 50
      }));
      
      // Only keep last 90 days
      if (processedData.sentimentTimeline.length > 90) {
        processedData.sentimentTimeline = processedData.sentimentTimeline.slice(-90);
      }
    }
    
    // Process correlation matrix
    if (rawData.correlationMatrix) {
      processedData.correlationMatrix = {};
      
      rawData.correlationMatrix.forEach(row => {
        const strategy = row[''] || Object.values(row)[0];
        processedData.correlationMatrix[strategy] = {};
        
        Object.keys(row).forEach(col => {
          if (col !== '' && col !== strategy) {
            processedData.correlationMatrix[strategy][col] = parseFloat(row[col]) || 0;
          }
        });
      });
    }

    return processedData;
  }

  parseValidationMetrics(metricsText) {
    const models = {};
    const lines = metricsText.split('\n').filter(line => line.trim());
    
    // Parse format: Model  Out-of-Sample R2  Directional Accuracy (%)
    lines.forEach(line => {
      const parts = line.trim().split(/\s{2,}/); // Split by multiple spaces
      if (parts.length >= 3 && parts[0] !== 'Model') {
        const modelName = parts[0].toLowerCase();
        const r2 = parseFloat(parts[1]) || 0;
        const dirAccuracy = parseFloat(parts[2].replace('%', '')) || 0;
        
        models[modelName] = {
          name: this.formatModelName(modelName),
          r2: r2,
          directionAccuracy: dirAccuracy
        };
      }
    });
    
    return models;
  }
  
  formatModelName(model) {
    const nameMap = {
      'ridge': 'Ridge Regression',
      'lightgbm': 'LightGBM',
      'ols': 'OLS',
      'ensemble': 'Ensemble Model'
    };
    return nameMap[model.toLowerCase()] || model;
  }

  formatStrategyName(strategy) {
    const nameMap = {
      'RIDGE': 'Ridge Regression',
      'BOOST': 'LightGBM Boost',
      'OLS': 'OLS',
      'ENSEMBLE': 'Ensemble Model',
      'EW': 'Equal Weight',
      'BTC': 'Bitcoin Only',
      'MV_OPT': 'AI-Enhanced Mean-Variance',
      'RISK_PARITY_ML': 'ML Risk Parity',
      'RISK_PARITY_EW': 'Risk Parity EW',
      'VOL_TARGET_EW': 'Volatility Targeted EW',
      'VOL_TARGET_ML': 'Volatility Targeted ML'
    };
    return nameMap[strategy] || strategy;
  }

  calculateAverage(data, field) {
    if (!data || data.length === 0) return 0;
    const sum = data.reduce((acc, row) => acc + (row[field] || 0), 0);
    return sum / data.length;
  }
}

export default new DataLoader();