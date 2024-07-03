import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.stats import norm


class StockAnalysis:
    def __init__(self, ticker: str):
        """
        Initialize the StockAnalysis object with a ticker symbol.

        :param ticker: The ticker symbol for the stock.
        """
        self.ticker = ticker
        self.data = None

    def fetch_data(self) -> None:
        """
        Fetch historical stock data for the given ticker and store it in the 'data' attribute.

        Uses the Yahoo Finance API to get the stock's historical price data over the maximum available period.
        """
        # Create a Ticker object using the given ticker symbol
        stock = yf.Ticker(self.ticker)

        # Fetch historical data for the maximum available period and store it in self.data
        self.data = stock.history(period="max")

    def calculate_macd(
        self, short_window: int = 12, long_window: int = 26, signal_window: int = 9
    ) -> None:
        """
        Calculate the Moving Average Convergence Divergence (MACD) and Signal Line for the stock.

        :param short_window: The span for the short-term EMA. Default is 12.
        :param long_window: The span for the long-term EMA. Default is 26.
        :param signal_window: The span for the Signal Line EMA. Default is 9.
        """
        # Calculate short-term EMA
        short_ema = self.data.Close.ewm(span=short_window, adjust=False).mean()

        # Calculate long-term EMA
        long_ema = self.data.Close.ewm(span=long_window, adjust=False).mean()

        # Calculate MACD
        self.data["MACD"] = short_ema - long_ema

        # Calculate Signal Line
        self.data["Signal_Line"] = self.data.MACD.ewm(
            span=signal_window, adjust=False
        ).mean()

    def calculate_normalized_macd(
        self, short_window: int = 12, long_window: int = 26, signal_window: int = 9
    ) -> None:
        """
        Calculate the normalized MACD and Signal Line for the stock.

        :param short_window: The span for the short-term EMA. Default is 12.
        :param long_window: The span for the long-term EMA. Default is 26.
        :param signal_window: The span for the Signal Line EMA. Default is 9.
        """
        # Calculate short-term EMA
        short_ema = self.data.Close.ewm(span=short_window, adjust=False).mean()

        # Calculate long-term EMA
        long_ema = self.data.Close.ewm(span=long_window, adjust=False).mean()

        # Calculate MACD
        self.data["MACD"] = short_ema - long_ema

        # Calculate Signal Line
        self.data["Signal_Line"] = self.data.MACD.ewm(
            span=signal_window, adjust=False
        ).mean()

        # Normalize MACD and Signal Line
        self.data["MACD"] = (self.data["MACD"] - self.data["MACD"].mean()) / self.data[
            "MACD"
        ].std()
        self.data["Signal_Line"] = (
            self.data["Signal_Line"] - self.data["Signal_Line"].mean()
        ) / self.data["Signal_Line"].std()

    def calculate_percentile_macd(
        self, short_window: int = 12, long_window: int = 26, signal_window: int = 9
    ) -> None:
        """
        Calculate the percentile-based MACD and Signal Line for the stock.

        :param short_window: The span for the short-term EMA. Default is 12.
        :param long_window: The span for the long-term EMA. Default is 26.
        :param signal_window: The span for the Signal Line EMA. Default is 9.
        """
        # Calculate short-term EMA
        short_ema = self.data.Close.ewm(span=short_window, adjust=False).mean()

        # Calculate long-term EMA
        long_ema = self.data.Close.ewm(span=long_window, adjust=False).mean()

        # Calculate MACD
        self.data["MACD"] = short_ema - long_ema

        # Calculate Signal Line
        self.data["Signal_Line"] = self.data.MACD.ewm(
            span=signal_window, adjust=False
        ).mean()

        # Normalize MACD and Signal Line
        self.data["MACD"] = (self.data["MACD"] - self.data["MACD"].mean()) / self.data[
            "MACD"
        ].std()
        self.data["Signal_Line"] = (
            self.data["Signal_Line"] - self.data["Signal_Line"].mean()
        ) / self.data["Signal_Line"].std()

        # Calculate MACD and Signal Line percentiles
        self.data["MACD"] = norm.cdf(self.data["MACD"]) * 200 - 100
        self.data["Signal_Line"] = norm.cdf(self.data["Signal_Line"]) * 200 - 100

    def find_crossovers(
        self, bullish_threshold: float, bearish_threshold: float
    ) -> None:
        """
        Identify bullish and bearish crossovers based on MACD and Signal Line.

        :param bullish_threshold: The maximum value for the Signal Line to classify a crossover as bullish.
        :param bearish_threshold: The minimum value for the Signal Line to classify a crossover as bearish.
        """
        self.data["Crossover"] = 0

        # Identify bullish crossovers
        bullish_indices = self.data.index[
            (self.data["MACD"] > self.data["Signal_Line"])
            & (self.data["MACD"].shift() < self.data["Signal_Line"].shift())
            & (self.data["Signal_Line"] < bullish_threshold)
        ]
        self.data.loc[bullish_indices, "Crossover"] = 1

        # Identify bearish crossovers
        bearish_indices = self.data.index[
            (self.data["MACD"] < self.data["Signal_Line"])
            & (self.data["MACD"].shift() > self.data["Signal_Line"].shift())
            & (self.data["Signal_Line"] > bearish_threshold)
        ]
        self.data.loc[bearish_indices, "Crossover"] = -1

    def get_fundamentals(self):
        """
        Fetch fundamental financial statements of the company.

        Returns the income statement, balance sheet, and cash flow statement of the company.
        """
        stock = yf.Ticker(self.ticker)
        return stock.income_stmt, stock.balance_sheet, stock.cashflow

    def create_fig(self) -> go.Figure:
        """
        Create a Plotly figure with candlestick chart, calculated moving averages,
        MACD, Signal Line, and marked crossovers.

        :return: A Plotly Figure object.
        """
        # Calculate various moving averages
        self.data["MA12"] = self.data["Close"].rolling(window=12).mean()
        self.data["MA26"] = self.data["Close"].rolling(window=26).mean()
        self.data["MA50"] = self.data["Close"].rolling(window=50).mean()
        self.data["MA200"] = self.data["Close"].rolling(window=200).mean()

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=(f"{self.ticker} Candlestick", "MACD"),
            row_width=[0.2, 0.7],
        )

        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data["Open"],
                high=self.data["High"],
                low=self.data["Low"],
                close=self.data["Close"],
                name="Candlestick",
            ),
            row=1,
            col=1,
        )

        for ma, color in zip(
            ["MA12", "MA26", "MA50", "MA200"], ["magenta", "cyan", "yellow", "black"]
        ):
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data[ma],
                    line=dict(color=color, width=1.5),
                    name=f"{ma} days MA",
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data["MACD"],
                line=dict(color="blue", width=2),
                name="MACD",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data["Signal_Line"],
                line=dict(color="orange", width=2),
                name="Signal Line",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=self.data[self.data["Crossover"] == 1].index,
                y=self.data[self.data["Crossover"] == 1]["MACD"],
                marker_symbol="triangle-up",
                marker_color="green",
                marker_size=20,
                name="Bullish Crossover (MACD) ✅",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=self.data[self.data["Crossover"] == -1].index,
                y=self.data[self.data["Crossover"] == -1]["MACD"],
                marker_symbol="triangle-down",
                marker_color="red",
                marker_size=20,
                name="Bearish Crossover (MACD) 🈲",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=self.data[self.data["Crossover"] == 1].index,
                y=self.data[self.data["Crossover"] == 1]["Close"],
                marker_symbol="triangle-up",
                marker_color="green",
                marker_size=25,
                name="Bullish Crossover (Close) ✅",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=self.data[self.data["Crossover"] == -1].index,
                y=self.data[self.data["Crossover"] == -1]["Close"],
                marker_symbol="triangle-down",
                marker_color="red",
                marker_size=25,
                name="Bearish Crossover (Close) 🈲",
            ),
            row=1,
            col=1,
        )

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=800,
        )

        return fig

    def generate_simulated_data(self, num_days: int) -> pd.DataFrame:
        """
        Generate simulated future stock data based on historical means and standard deviations.

        :param num_days: The number of days to simulate.
        :return: A DataFrame containing the simulated stock data.
        """
        # Get historical means and standard deviations
        means = self.data.mean()
        stds = self.data.std()

        # Create a DataFrame to hold random returns
        random_returns = pd.DataFrame()

        # Generate random returns for each column in the historical data
        for col in self.data.columns:
            random_returns[col] = np.random.normal(
                loc=means[col], scale=stds[col], size=num_days
            )

        # Convert random returns to cumulative product of factors
        random_returns += 1
        factors = random_returns.cumprod()

        # Create future dates for the simulation
        last_date = self.data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(days=1), periods=num_days
        )

        # Create a DataFrame with future data
        future_data = pd.DataFrame(
            index=future_dates, columns=self.data.columns, data=factors.values
        )

        # Combine historical and future data into one DataFrame
        simulated_data = pd.concat([self.data, future_data])

        return simulated_data
