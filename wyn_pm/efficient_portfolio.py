from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
import plotly.graph_objects as go
import yfinance as yf


class EfficientPortfolio:
    def __init__(
        self, tickers: List[str], start_date: str, end_date: str, interval: str
    ):
        """
        Initialize the EfficientPortfolio class with tickers, start and end dates, and interval.

        Args:
            tickers (List[str]): List of stock ticker symbols.
            start_date (str): Start date for data retrieval in 'YYYY-MM-DD' format.
            end_date (str): End date for data retrieval in 'YYYY-MM-DD' format.
            interval (str): Data interval (e.g., '1d', '1wk', '1mo').
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.data = None

    def download_stock_data(self) -> pd.DataFrame:
        """
        Download stock data for given tickers between start_date and end_date.

        Returns:
            pd.DataFrame: DataFrame with adjusted close prices for the given tickers.
        """
        self.data = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
        )["Adj Close"]
        return self.data

    def download_stocks(self) -> List[pd.DataFrame]:
        """
        Downloads stock data from Yahoo Finance.

        Returns:
            List[pd.DataFrame]: A list of Pandas DataFrames, one for each stock.
        """
        df_list = []
        for ticker in self.tickers:
            df = yf.download(ticker)
            df_list.append(df.tail(255 * 8))
        return df_list

    def create_portfolio_and_calculate_returns(self, top_n: int) -> pd.DataFrame:
        """
        Create a portfolio and calculate returns based on the given top N stocks.

        Args:
            top_n (int): Number of top stocks to include in the portfolio.

        Returns:
            pd.DataFrame: DataFrame containing calculated returns and portfolio history.
        """
        returns_data = self.data.pct_change()
        returns_data.dropna(inplace=True)

        portfolio_history = []
        portfolio_returns = []

        # Loop over the data in window_size-day windows
        window_size = 1
        for start in range(0, len(returns_data) - window_size, window_size):
            end = start + window_size
            current_window = returns_data[start:end]
            top_stocks = (
                current_window.mean()
                .sort_values(ascending=False)
                .head(top_n)
                .index.tolist()
            )
            next_window = returns_data[end : end + window_size][top_stocks].mean(axis=1)

            portfolio_returns.extend(next_window)
            added_length = len(next_window)
            portfolio_history.extend([top_stocks] * added_length)

        new_returns_data = returns_data.copy()
        new_returns_data = new_returns_data.iloc[0:-window_size, :]
        new_returns_data["benchmark"] = new_returns_data.apply(
            lambda x: x[0:5].mean(), axis=1
        )
        new_returns_data["portfolio_returns"] = portfolio_returns
        new_returns_data["portfolio_history"] = portfolio_history
        new_returns_data["rolling_benchmark"] = (
            new_returns_data["benchmark"] + 1
        ).cumprod()
        new_returns_data["rolling_portfolio_returns"] = (
            new_returns_data["portfolio_returns"] + 1
        ).cumprod()

        return new_returns_data

    @staticmethod
    def portfolio_annualised_performance(
        weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute the annualized performance of the portfolio in terms of its standard deviation (volatility) and expected returns.

        Args:
            weights (np.ndarray): The weights of the assets in the portfolio.
            mean_returns (np.ndarray): The mean (expected) returns of the assets.
            cov_matrix (np.ndarray): The covariance matrix of the asset returns.

        Returns:
            Tuple[float, float]: Tuple of portfolio volatility (standard deviation) and portfolio expected return, both annualized.
        """
        returns = np.sum(mean_returns * weights) * 252
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return std, returns

    @staticmethod
    def random_portfolios(
        num_portfolios: int,
        num_weights: int,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Generate random portfolios and calculate their standard deviation, returns and Sharpe ratio.

        Args:
            num_portfolios (int): The number of random portfolios to generate.
            mean_returns (np.ndarray): The mean (expected) returns of the assets.
            cov_matrix (np.ndarray): The covariance matrix of the asset returns.
            risk_free_rate (float): The risk-free rate of return.

        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: Tuple of results and weights_record.
                results (np.ndarray): A 3D array with standard deviation, returns and Sharpe ratio of the portfolios.
                weights_record (List[np.ndarray]): A list with the weights of the assets in each portfolio.
        """
        results = np.zeros((3, num_portfolios))
        weights_record = []

        for i in np.arange(num_portfolios):
            weights = np.random.random(num_weights)
            weights /= np.sum(weights)
            weights_record.append(weights)
            portfolio_std_dev, portfolio_return = (
                EfficientPortfolio.portfolio_annualised_performance(
                    weights, mean_returns, cov_matrix
                )
            )
            results[0, i] = portfolio_std_dev
            results[1, i] = portfolio_return
            results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev

        return results, weights_record

    def display_simulated_ef_with_random(
        self,
        mean_returns: List[float],
        cov_matrix: np.ndarray,
        num_portfolios: int,
        risk_free_rate: float,
    ) -> plt.Figure:
        """
        Display a simulated efficient frontier plot based on randomly generated portfolios.

        Args:
            mean_returns (List[float]): A list of mean returns for each security or asset in the portfolio.
            cov_matrix (np.ndarray): A covariance matrix for the securities or assets in the portfolio.
            num_portfolios (int): The number of random portfolios to generate.
            risk_free_rate (float): The risk-free rate of return.

        Returns:
            plt.Figure: A pyplot figure object.
        """
        results, weights = self.random_portfolios(
            num_portfolios, len(mean_returns), mean_returns, cov_matrix, risk_free_rate
        )

        max_sharpe_idx = np.argmax(results[2])
        sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
        max_sharpe_allocation = pd.DataFrame(
            weights[max_sharpe_idx], index=self.tickers, columns=["allocation"]
        )
        max_sharpe_allocation.allocation = [
            round(i * 100, 2) for i in max_sharpe_allocation.allocation
        ]
        max_sharpe_allocation = max_sharpe_allocation.T

        min_vol_idx = np.argmin(results[0])
        sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
        min_vol_allocation = pd.DataFrame(
            weights[min_vol_idx], index=self.tickers, columns=["allocation"]
        )
        min_vol_allocation.allocation = [
            round(i * 100, 2) for i in min_vol_allocation.allocation
        ]
        min_vol_allocation = min_vol_allocation.T

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(
            results[0, :],
            results[1, :],
            c=results[2, :],
            cmap="YlGnBu",
            marker="o",
            s=10,
            alpha=0.3,
        )
        ax.scatter(sdp, rp, marker="*", color="r", s=500, label="Maximum Sharpe ratio")
        ax.scatter(
            sdp_min, rp_min, marker="*", color="g", s=500, label="Minimum volatility"
        )
        ax.set_title("Simulated Portfolio Optimization based on Efficient Frontier")
        ax.set_xlabel("Annual volatility")
        ax.set_ylabel("Annual returns")
        ax.legend(labelspacing=0.8)

        details = {
            "Annualised Return": round(rp, 2),
            "Annualised Volatility": round(sdp, 2),
            "Max Sharpe Allocation": max_sharpe_allocation,
            "Max Sharpe Allocation in Percentile": max_sharpe_allocation.div(
                max_sharpe_allocation.sum(axis=1), axis=0
            ),
            "Min Volatility Allocation": min_vol_allocation,
            "Min Volatility Allocation in Percentile": min_vol_allocation.div(
                min_vol_allocation.sum(axis=1), axis=0
            ),
        }

        keys = []
        values = []

        for key, value in details.items():
            keys.append(key)
            values.append(value)

        table = PrettyTable()
        table.field_names = ["Key", "Value"]
        for i in range(len(keys)):
            table.add_row([keys[i], values[i]])

        print(table)

        return fig, details

    def plot_portfolio_performance(
        self, portfolio_returns: pd.DataFrame, height_of_graph: int
    ) -> go.Figure:
        """
        Plot the portfolio performance with benchmark and portfolio returns, and display rolling Sharpe ratios.

        Args:
            portfolio_returns (pd.DataFrame): DataFrame containing the portfolio returns data.
            height_of_graph (int): Height of the plot.

        Returns:
            go.Figure: A Plotly figure object showing the performance plot.
        """
        returns_data = portfolio_returns
        benchmark_sharpe_ratio = (
            returns_data["benchmark"].mean() / returns_data["benchmark"].std()
        )
        portfolio_sharpe_ratio = (
            returns_data["portfolio_returns"].mean()
            / returns_data["portfolio_returns"].std()
        )

        # Data for plotting
        df = returns_data[
            ["rolling_benchmark", "rolling_portfolio_returns", "portfolio_history"]
        ]
        df.index = pd.to_datetime(df.index, unit="ms")

        # Create plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["rolling_benchmark"],
                mode="lines",
                name="Rolling Benchmark",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["rolling_portfolio_returns"],
                mode="lines",
                name="Rolling Portfolio Returns",
            )
        )

        for date, stocks in df["portfolio_history"].items():
            fig.add_shape(
                type="line",
                x0=date,
                y0=0,
                x1=date,
                y1=0,
                line=dict(color="RoyalBlue", width=1),
            )
            fig.add_annotation(
                x=date,
                y=0.5,
                text=str(stocks),
                showarrow=False,
                yshift=10,
                textangle=-90,
                font=dict(size=15),  # You can adjust the size as needed
            )

        # Calculate means and standard deviations
        benchmark_mean = returns_data["benchmark"].mean()
        benchmark_std = returns_data["benchmark"].std()
        portfolio_mean = returns_data["portfolio_returns"].mean()
        portfolio_std = returns_data["portfolio_returns"].std()

        # Update title text with additional information
        if self.interval == "1mo":
            some_n_based_on_time_frame = 12
            in_a_year = 1000 * (1 + portfolio_mean) ** (some_n_based_on_time_frame)
            in_50_years = 1000 * (1 + portfolio_mean) ** (
                some_n_based_on_time_frame * 50
            )
        else:
            some_n_based_on_time_frame = 4
            in_a_year = 1000 * (1 + portfolio_mean) ** (some_n_based_on_time_frame)
            in_50_years = 1000 * (1 + portfolio_mean) ** (
                some_n_based_on_time_frame * 50
            )
        title_text = (
            f"Performance:<br>"
            f"Benchmark Sharpe Ratio = {benchmark_sharpe_ratio:.3f}, "
            f"Portfolio Sharpe Ratio = {portfolio_sharpe_ratio:.3f}, "
            f"based on interval: {self.interval}<br>"
            f"Benchmark => Mean: {benchmark_mean:.4f}, Std: {benchmark_std:.4f}; "
            f"Portfolio => Mean: {portfolio_mean:.4f}, Std: {portfolio_std:.4f}<br>"
            f"---<br>"
            f"This may or may not be a small number, let's check under the following cases 1) 12-mon, and 2) 50-year returns on $1,000 USD: <br>"
            f"1,000*(1+{portfolio_mean:.4f})^({some_n_based_on_time_frame})={in_a_year} in USD, <br>"
            f"1,000*(1+{portfolio_mean:.4f})^({some_n_based_on_time_frame}*50)={in_50_years} in USD."
        )
        curr_max_num = max(
            df.rolling_benchmark.max(), df.rolling_portfolio_returns.max()
        )
        fig.update_layout(
            title=title_text,
            xaxis_title="Date",
            yaxis_title="Value",
            yaxis=dict(range=[0, curr_max_num * 1.1]),
            legend=dict(
                orientation="h", x=0.5, y=-0.4, xanchor="center", yanchor="bottom"
            ),
            height=height_of_graph,
        )

        return fig
