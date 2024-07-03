# WYN-PM 📈💼

Welcome to the `wyn-pm` library, an official library from [W.Y.N. Associates, LLC](https://wyn-associates.com/) FinTech branch. This library provides tools for stock analysis, efficient portfolio generation, and training sequential neural networks for financial data.

[![YouTube Video](https://img.youtube.com/vi/85Rv7jg8gc8/0.jpg)](https://www.youtube.com/embed/85Rv7jg8gc8?si=L4zlKGmdJOu6bO82)

## Links

- [W.Y.N. Associates Fintech Library](https://wyn-associates.com/fintech/)
- [WYN PM on PyPI](https://pypi.org/project/wyn-pm/)
- [Web App](https://huggingface.co/spaces/eagle0504/Momentum-Strategy-Screener)
- [Github](https://github.com/yiqiao-yin/wyn-pm)

## Installation 🚀

To install the library, use the following command:

```bash
! pip install wyn-pm
```

Please feel free to use [this jupyter notebook](https://github.com/yiqiao-yin/WYNAssociates/blob/main/docs/ref-deeplearning/ex_%20-%20wyn-pm%20tutorial.ipynb) as reference.

## Stock Analyzer: Plot Buy/Sell Signal 📊

Analyze stocks and plot buy/sell signals using the MACD indicator.

### Example Usage:

```python
from wyn_pm.stock_analyzer import *

# Initialize stock analysis for a given ticker
stock_analysis = StockAnalysis(ticker="AAPL")

# Fetch stock data
stock_analysis.fetch_data()

# Calculate MACD
stock_analysis.calculate_macd()

# Find crossovers to generate buy/sell signals
stock_analysis.find_crossovers(bullish_threshold=-2, bearish_threshold=2)

# Create and show the plot
fig = stock_analysis.create_fig()
fig.show()
```

## Efficient Portfolio: Generate Optimal Weights 💹

Create an optimal portfolio by generating efficient weights for a list of stock tickers.

### Example Usage:

```python
from wyn_pm.efficient_portfolio import *

# Initialize portfolio with given tickers and date range
portfolio = EfficientPortfolio(tickers=["AAPL", "MSFT", "GOOGL"], start_date="2020-01-01", end_date="2022-01-01", interval="1d")

# Download stock data
stock_data = portfolio.download_stock_data()

# Calculate portfolio returns
portfolio_returns = portfolio.create_portfolio_and_calculate_returns(top_n=5)

# Calculate mean returns and covariance matrix
mean_returns = stock_data.pct_change().mean()
cov_matrix = stock_data.pct_change().cov()

# Define the number of portfolios to simulate and the risk-free rate
num_portfolios = 10000
risk_free_rate = 0.01

# Display the efficient frontier with randomly generated portfolios
fig, details = portfolio.display_simulated_ef_with_random(mean_returns.values, cov_matrix.values, num_portfolios, risk_free_rate)
fig.show()

# Print details of the max Sharpe and min volatility portfolios
print(details)
```

## Momentum Strategy: Generate Portfolio Arbitrage

Create a portfolio based on the famous **momentum strateg** in asset pricing given a list of stock tickers.

### Example Usage:

```python
# Acquire data for the "Momentum Strategy":
portfolio = EfficientPortfolio(tickers=["AAPL", "MSFT", "GOOGL", "NFLX", "IBM"], start_date="2017-01-01", end_date="2024-07-01", interval="1mo")
stock_data = portfolio.download_stock_data()
portfolio_returns = portfolio.create_portfolio_and_calculate_returns(top_n=3)

# Plot
fig = portfolio.plot_portfolio_performance(portfolio_returns, height_of_graph=600)
fig.show()
```

## Training Sequential Neural Networks: Stock Prediction 🤖📈

Train various neural network models on stock data and perform Monte Carlo simulations.

### Example Usage:

```python
from wyn_pm.trainer import *

# Example usage:
stock_modeling = StockModeling()

# Training: ~ 9 min on CPU
forecast_results, mc_figure = stock_modeling.forecast_and_plot(stock="AAPL", start_date="2020-01-01", end_date="2023-01-01", look_back=50, num_of_epochs=10, n_futures=365, n_samples=1000, verbose_style=1)

# Results
print(forecast_results)
mc_figure.show()
```

## Technical Discussion of the Momentum Strategy

### Monthly Momentum Factor (MOM)

The Monthly Momentum Factor (MOM) can be calculated by subtracting the equal-weighted average of the lowest performing firms from the equal-weighted average of the highest performing firms, lagged one month (Carhart, 1997). A stock exhibits momentum if its prior 12-month average of returns is positive. Similar to the three-factor model, the momentum factor is defined by a self-financing portfolio of (long positive momentum) + (short negative momentum). Momentum strategies remain popular in financial markets, and financial analysts often incorporate the 52-week price high/low in their Buy/Sell recommendations.

- Carhart, M. M. (1997). On persistence in mutual fund performance. The Journal of finance, 52(1), 57-82. [link](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1997.tb03808.x)

### Four-Factor Model

The four-factor model is commonly used for active management and mutual fund evaluation. Three commonly used methods to adjust a mutual fund's returns for risk are:

#### 1. Market Model:

$$
EXR_t = \alpha^J + \beta_{mkt} * EXMKT_t + \epsilon_t
$$
The intercept in this model is referred to as "Jensen's alpha".

- The Valuation of Risk Assets and the Selection of Risky Investments in Stock Portfolios and Capital Budgets, [link](https://www.sciencedirect.com/science/article/abs/pii/B9780127808505500186)
- Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk, [link](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1964.tb02865.x)

### 2. Fama–French Three-Factor Model:

$$
EXR_t = \alpha^{FF} + \beta_{mkt} * EXMKT_t + \beta_{HML} * HML_t + \beta_{SMB} * SMB_t + \epsilon_t
$$
The intercept in this model is referred to as the "three-factor alpha".

- Common risk factors in the returns on stocks and bonds, [link](https://www.sciencedirect.com/science/article/abs/pii/0304405X93900235?via%3Dihub)
- The Capital Asset Pricing Model: Theory and Evidence, [link](https://mba.tuck.dartmouth.edu/bespeneckbo/default/AFA611-Eckbo%20web%20site/AFA611-S6B-FamaFrench-CAPM-JEP04.pdf)
- The other side of value: The gross profitability premium, [link](https://www.sciencedirect.com/science/article/abs/pii/S0304405X13000044)
- A five-factor asset pricing model, [link](https://www.sciencedirect.com/science/article/abs/pii/S0304405X14002323?via%3Dihub)

### 3. Carhart Four-Factor Model:

$$
EXR_t = \alpha^c + \beta_{mkt} * EXMKT_t + \beta_{HML} * HML_t + \beta_{SMB} * SMB_t + \beta_{UMD} * UMD_t + \epsilon_t
$$

The intercept in this model is referred to as the "four-factor alpha". `EXR_t` is the monthly return to the asset of concern in excess of the monthly t-bill rate. These models are used to adjust for risk by regressing the excess returns of the asset on an intercept (the alpha) and some factors on the right-hand side of the equation that attempt to control for market-wide risk factors. The right-hand side risk factors include the monthly return of the CRSP value-weighted index less the risk-free rate (`EXMKT_t`), monthly premium of the book-to-market factor (`HML_t`), monthly premium of the size factor (`SMB_t`), and the monthly premium on winners minus losers (`UMD_t`) from Fama-French (1993) and Carhart (1997).

A fund manager demonstrates forecasting ability when their fund has a positive and statistically significant alpha.

SMB is a zero-investment portfolio that is long on small capitalization (cap) stocks and short on big-cap stocks. Similarly, HML is a zero-investment portfolio that is long on high book-to-market (B/M) stocks and short on low B/M stocks, and UMD is a zero-cost portfolio that is long previous 12-month return winners and short previous 12-month loser stocks.

---

Enjoy analyzing stocks, creating efficient portfolios, and training neural networks with `wyn-pm`! If you have any questions, feel free to reach out.

Happy coding! 🖥️✨