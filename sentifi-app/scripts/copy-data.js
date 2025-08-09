// scripts/copy-data.js
const fs = require('fs-extra');
const path = require('path');

// 修正路径 - 需要往上走两级
const sourceDir = path.resolve(__dirname, '../../Stage3/results');
const targetDir = path.resolve(__dirname, '../public/data');
const targetImgDir = path.resolve(__dirname, '../public/images/sentiment');

try {
  // 检查源目录是否存在
  if (!fs.existsSync(sourceDir)) {
    console.error('❌ Source directory not found:', sourceDir);
    console.error('Please make sure Stage3/results exists');
    process.exit(1);
  }

  fs.ensureDirSync(targetDir);
  fs.emptyDirSync(targetDir);
  
  fs.ensureDirSync(targetImgDir);
  
  const sentimentImgs = [
    'figures/sentiment/fear_greed_gauge_.png',
    'figures/sentiment/hist_final.png',
    'figures/causality/granger_ret_btc_to_d_Overall.png',
    'figures/causality/granger_d_Overall_to_ret_btc.png',
    
  ];
  
  sentimentImgs.forEach(img => {
    const sourcePath = path.join(sourceDir, img);
    const targetPath = path.join(targetImgDir, path.basename(img));
    
    if (fs.existsSync(sourcePath)) {
      fs.copySync(sourcePath, targetPath);
      console.log(`✅ Copied image: ${img}`);
    }
  });

  // 复制特定文件
  const filesToCopy = [
    'integrated_results/integrated_portfolios.csv',
    
    'integrated_results/cumulative_returns_data.csv',
    'integrated_results/cumulative_log_returns_data.csv',
    'integrated_results/monthly_returns_data.csv',
    'integrated_results/correlation_matrix_data.csv',
    
    'integrated_results/drawdown_timeline_data.csv',
    'integrated_results/ml_predictions_integrated.csv',
    'integrated_results/integrated_features.csv',
    'news_data/compound_timeseries_daily.csv',
    'figures/metrics_table_sorted.csv',
    'figures/statistical_validation_metrics.txt'
  ];

  filesToCopy.forEach(file => {
    const sourcePath = path.join(sourceDir, file);
    const targetPath = path.join(targetDir, path.basename(file));
    
    if (fs.existsSync(sourcePath)) {
      fs.copySync(sourcePath, targetPath);
      console.log(`✅ Copied: ${file}`);
    } else {
      console.warn(`⚠️  File not found: ${file}`);
    }
  });

  console.log('✅ Data files copied successfully!');

} catch (err) {
  console.error('❌ Error copying data files:', err);
}