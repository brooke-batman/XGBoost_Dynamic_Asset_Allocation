# Dynamic Asset Allocation via Machine Learning & Macro Regime Detection

## Executive Summary
This project develops an end-to-end Machine Learning pipeline utilizing an **XGBoost regime-switching classifier** to solve a critical institutional problem: dynamically adjusting factor allocations across shifting macroeconomic climates. By training a multi-class model on key macroeconomic features (including the Real Policy Rate, VIX Volatility Shock, and Credit Spread Velocity), the strategy shifts capital dynamically among growth equities, low-volatility stocks, gold, and fixed-income safe havens. The out-of-sample backtest (2022–2025) demonstrates that this machine learning strategy successfully mitigates downside risk during market shocks while delivering a **19.31% Annualized Return** and a **1.78 Sharpe Ratio**, significantly outperforming a static 60/40 benchmark.

## Business Problem & Objectives
Traditional static allocation models, such as the classic 60/40 stock/bond portfolio, rely on historical negative correlations between equities and fixed-income assets to cushion downside risk. However, in regimes characterized by inflationary shocks and rising interest rates (such as 2022), both stocks and bonds can experience simultaneous sell-offs, breaking down traditional diversification.

**Key Objectives:**
- **Proactive Risk Mitigation:** Classify market states into distinct economic regimes (Normal Growth, Elevated Macro Risk, and Panic/Crash Shocks) using forward-looking indicators.
- **Dynamic Rotation:** Shift capital out of high-beta equity factors and into safe havens or low-volatility factors before market drawdowns accelerate.
- **Risk-Adjusted Outperformance:** Outperform the static 60/40 benchmark, an equal-weight portfolio, and a raw buy-and-hold S&P 500 index on a risk-adjusted basis (maximizing Sharpe and Sortino ratios while minimizing Maximum Drawdown).

## Methodology & Data

### 1. Data Ingestion & Features
Data was fetched programmatically from Yahoo Finance (ETF pricing) and the St. Louis FRED API (Macroeconomic Indicators) spanning 2015 to 2025. Key engineered features include:
- **Real Policy Rate:** Federal Funds Rate minus YoY CPI Inflation (anti-leakage aligned).
- **VIX Volatility Shock Ratio:** 10-day SMA of the VIX divided by its 200-day SMA to identify sudden volatility spikes.
- **Credit Spread Velocity:** 10-day percentage rate of change in the high-yield corporate credit spread.
- **Stock-Bond EWMA Correlation:** Dynamic rolling 30-day exponential window tracking asset-class decoupling.
- **Downside Volatility Ratio:** The ratio of 20-day downside deviations to upside deviations in the equity market.

### 2. Model Architecture & Training
- **Model:** Multiclass XGBoost Classifier (`XGBClassifier`).
- **Target Regimes:** Defined programmatically using a 10-day forward-looking risk horizon:
  - **Regime 0 (Normal Growth):** Forward returns are positive, and volatility is low.
  - **Regime 1 (Elevated Macro Risk):** Returns are stagnant or volatility is elevated.
  - **Regime 2 (Panic/Crash Shock):** Forward 10-day drawdown exceeds -2.5%.
- **Validation:** Chronological train/test split (70% Train: 2016-2022 / 30% Out-of-Sample Test: 2022-2025) to prevent chronological leakage.
- **Class Imbalance Mitigation:** Balanced sample weights adjusted during training.

---

## Key Results & Strategy Performance

The out-of-sample backtest evaluations (December 2022 – December 2025) confirm the strategy's strong performance:

| Strategy | Annualized Return | Annual Volatility | Sharpe Ratio (Rf=4%) | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost Tactical Strategy** | **19.31%** | **10.82%** | **1.41** | **-12.54%** |
| **Benchmark 1: Static 60/40** | 14.28% | 9.79% | 1.05 | -11.41% |
| **Benchmark 2: Equal-Weight (1/N)** | 18.10% | 11.20% | 1.26 | -13.37% |
| **Benchmark 3: S&P 500 (SPY)** | 20.90% | 15.30% | 1.10 | -19.21% |

### Core Performance Insights:
- **Downside Protection:** During periods flagged as *Regime 2 (Crash)*, the model dynamically shifted 100% of the portfolio into safe-haven Treasury bonds (`AGG`), successfully capping the maximum drawdown to just **-12.54%** (compared to the S&P 500's drawdown of **-19.21%**).
- **Superior Efficiency:** The strategy achieved an institutional-grade Sharpe Ratio of **1.41**, outperforming the static 60/40 portfolio's Sharpe ratio of **1.05**.

---

## Repository Structure
```text
Dynamic-Factor-Allocation-XGBoost/
├── Data/
│   ├── institutional_performance_matrix.csv  # Unified strategy vs. benchmark metrics
│   └── portfolio_results.csv                 # Daily return timeseries & predicted regimes
├── Notebooks/
│   └── Dynamic_Factor_Allocation_XGBoost.ipynb
└── README.md
```

----

## How to Run & Reproduce
```text
1. Prerequisites & Environment Setup:
git clone [https://github.com/brooke-batman/XGBoost-Dynamic-Asset-Allocation.git](https://github.com/brooke-batman/XGBoost-Dynamic-Asset-Allocation.git)
cd XGBoost-Dynamic-Asset-Allocation
pip install pandas numpy scikit-learn xgboost yfinance matplotlib seaborn

2. Execution:
- Open Jupyter Notebook, JupyterLab, or Google Colab.
- Navigate to Notebook/ and open Dynamic_Asset_Allocation_Project.ipynb.
- Select Kernel -> Restart & Run All to execute data fetching, model training, and backtesting.

