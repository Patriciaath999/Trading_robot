# Taiwan Stock Trading: Deep RL vs Technical Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comparative study of Deep Reinforcement Learning (PPO) and Traditional Technical Analysis strategies for Taiwan stock market trading.

## 🎯 Project Overview

This project compares the performance of deep reinforcement learning methods against traditional technical analysis in Taiwan stock trading. The experiment includes:

- **Two versions of PPO (Proximal Policy Optimization)** models
- **One technical analysis baseline** strategy  
- **Manual trading** strategy for reference
- **Backtesting** on 26 Taiwan stocks over 3 months

**Key Finding**: In short-term bull markets, simple technical analysis (11.05% return) significantly outperforms deep learning approaches (0.56% - 1.79% return).

## 📊 Performance Summary

| Strategy | Return (3M) | Annualized Est. | Training Time | Win Rate | Complexity |
|----------|-------------|-----------------|---------------|----------|------------|
| **Manual Trading** | **13.44%** | **53.8%** | - | 90.9% | Low |
| **Technical Analysis** | **11.05%** | **44.2%** | 0.2s | - | Low |
| Simplified PPO | 1.79% | 7.2% | 11 hours | - | High |
| Original PPO | 0.56% | 2.2% | 4 hours | - | High |

**Test Period**: 2025/09/09 - 2026/01/09 (3 months, bull market)  
**Initial Capital**: NT$ 100,000,000  
**Stocks**: 26 Taiwan stocks

## 🔍 Key Insights

### Why Deep Learning Underperformed

1. **Action Space Explosion**
   - Discrete actions: 3^26 = 2.5 trillion combinations
   - 500k training steps explored only 0.00002% of possibilities
   - Model converged to overly conservative "safe" strategy

2. **Overly Conservative Strategy**
   - Reward function penalized volatility (-0.5×volatility)
   - Model learned to avoid actions → missed bull market opportunities
   - Only invested 5% of capital per trade vs 20% for technical analysis

3. **Training-Test Distribution Mismatch**
   - Trained on mixed market conditions (2021-2024)
   - Tested on pure bull market (2025)
   - Learned "stability in uncertainty" instead of "aggression in bull market"

4. **High Noise-to-Signal Ratio**
   - Daily returns too noisy for effective learning
   - Model couldn't distinguish skill from luck
   - Technical indicators already captured market patterns

### When Deep Learning DOES Work

Based on literature review (Ensemble Strategy 2016-2020):

- ✅ **Long-term trading** (5+ years, 65% total return over 4.5 years)
- ✅ **Rolling retraining** (retrain every 3 months with latest data)
- ✅ **Model ensemble** (A2C + PPO + DDPG competition)
- ✅ **Mixed market conditions** (bull + bear + sideways)
- ❌ **Short-term bull market** (this study - simple rules work better)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- TA-Lib library
- CUDA (optional, for GPU training)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stock-trading-rl-analysis.git
cd stock-trading-rl-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Run

```bash
# Run technical analysis strategy (recommended)
python models/technical_analysis.py

# Train simplified PPO (takes ~11 hours)
python models/ppo_simplified.py

# Visualize results
python scripts/visualize_results.py
```

## 📈 Methodology

### Models Tested

#### 1. Original PPO
**Architecture**:
- State Space: 391 dimensions (1 + 26 stocks × 15 indicators)
- Action Space: Continuous [-1, 1]^26
- Reward: Daily portfolio return

**Technical Indicators** (15):
MACD, RSI, CCI, ADX, Momentum, ROC, Close-to-SMA, MACD Histogram, Bollinger Bands (Position & Width), Volatility, Volume Ratio, ATR, OBV, Turbulence

**Training**:
- Period: 2016-2022 (7 years)
- Steps: 200,000
- Time: ~4 hours

**Result**: 0.56% return (failed to learn effective strategy)

#### 2. Simplified PPO

**Key Improvements**:
1. **Reduced Features**: 6 indicators (MACD, Momentum, RSI, Close-to-SMA, Volume Ratio, Turbulence)
2. **Discrete Actions**: {Sell=0, Hold=1, Buy=2}^26 instead of continuous
3. **Enhanced Reward**: `Return - 0.5×Volatility + 0.1×Diversity`
4. **Post-COVID Training**: 2021-2024 (4 years)

**Architecture**:
- State Space: 183 dimensions (53% reduction)
- Action Space: Discrete {0,1,2}^26
- Reward: Multi-objective (return, stability, diversification)

**Training**:
- Period: 2021-2024 (post-COVID)
- Steps: 500,000
- Time: ~11 hours

**Result**: 1.79% return (3.2× improvement, still underperformed)

#### 3. Technical Analysis (Baseline)

**Strategy**:
- **Buy Signal** (≥5 conditions met):
  - MACD uptrend (MACD > 0, histogram > 0)
  - Positive momentum
  - Price above 20-day SMA
  - RSI in healthy range (40-70)
  - Volume increase
  - Bollinger Bands position (0.2-0.8)

- **Sell Signal** (≥4 conditions met):
  - MACD downtrend
  - RSI overbought (>70)
  - Price below SMA
  - Negative momentum

**Execution**:
- Invest 20% of available cash per buy signal
- Sell all holdings on sell signal
- Execution time: 0.2 seconds

**Result**: 11.05% return, 83 trades

#### 4. Manual Trading (Actual Investment)

**Strategy**:
- Concentrated portfolio: 11 stocks
- Heavy allocation to TSMC (67.1%)
- Focus on quality tech stocks
- Buy-and-hold approach

**Result**: 13.44% return, 90.9% win rate

## 📂 Project Structure

```
stock-trading-rl-analysis/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── docs/                              # Documentation
│   ├── report.md                      # Complete analysis report
│   ├── ppo_analysis.md               # Detailed PPO analysis
│   └── images/                        # Figures and charts
│       ├── performance_comparison.png
│       └── portfolio_allocation.png
│
├── models/                            # Model implementations
│   ├── ppo_original.py               # Original PPO (391-dim state)
│   ├── ppo_simplified.py             # Simplified PPO (183-dim state)
│   ├── technical_analysis.py         # Multi-indicator strategy
│   └── environment.py                # Trading environment
│
├── results/                           # Experimental results
│   ├── ppo_original/
│   ├── ppo_simplified/
│   └── technical_analysis/
│
└── scripts/                           # Utility scripts
    ├── train_ppo.py                  # Training script
    ├── test_strategy.py              # Testing script
    └── visualize_results.py          # Visualization tools
```

## 🔬 Detailed Analysis

### Problem Diagnosis: Why PPO Failed

**Problem 1: State Space Too Large**
- 391 dimensions → overfitting risk
- Too much redundant information
- Solution: Reduced to 6 core indicators (↑220% improvement)

**Problem 2: Action Space Explosion**
- 3^26 = 2.5 trillion discrete combinations
- 500k steps ≈ 0.00002% exploration
- Model stuck in local optimum (conservative strategy)

**Problem 3: Reward Function Design**
- Volatility penalty discouraged profitable actions
- Bull markets naturally have higher volatility
- Model prioritized stability over returns

**Problem 4: Training-Test Mismatch**
- Trained on 2021-2024 (mixed conditions)
- Tested on 2025 (pure bull market)
- Learned wrong strategy for test environment

See [docs/report.md](docs/report.md) for complete analysis.

## 📚 References

1. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms". arXiv:1707.06347
2. Yang, H., et al. (2020). "Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy". ICAIF 2020
3. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning". Nature 518

## 💡 Recommendations

### For Practitioners

**Use Technical Analysis if**:
- ✅ Trading timeframe < 1 year
- ✅ Clear market trend (bull/bear)
- ✅ Limited computational resources
- ✅ Need explainable decisions

**Consider Deep RL if**:
- ✅ Trading timeframe > 5 years
- ✅ Can implement rolling retraining (< 30 min per retrain)
- ✅ Have ensemble infrastructure
- ✅ Complex multi-asset portfolio

### For Researchers

**Future Work**:
1. Implement rolling retraining every 1-3 months
2. Test ensemble methods (A2C + PPO + DDPG)
3. Reduce training time to < 30 minutes
4. Explore simpler action spaces (e.g., top-5 stocks only)
5. Test on longer periods (> 3 years)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

Areas for contribution:
- Implementation of rolling training
- Additional trading strategies
- Improved visualizations
- Extended backtesting periods

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Nancy**
- Graduate Student, Computer Science, National Cheng Kung University (NCKU)
- Previously: B.S. Agricultural Economics, National Taiwan University (NTU)
- Research Interests: Deep Reinforcement Learning, Algorithmic Trading, Financial AI

## 🙏 Acknowledgments

- **National Cheng Kung University** - Research support
- **Stable-Baselines3** - Reinforcement learning library
- **TA-Lib** - Technical analysis indicators
- **OpenAI Gym** - Environment framework

## ⚠️ Disclaimer

This project is for educational and research purposes only. It does not constitute financial advice. Trading stocks involves risk of loss. Past performance does not guarantee future results.

---

⭐ **If you find this project useful, please star it!**

📧 **Contact**: [Your Email] | [LinkedIn] | [Personal Website]
