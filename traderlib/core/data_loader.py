import pandas as pd
import yfinance as yf

class DataLoader:
    """
    A simple data loader using yfinance to fetch and preprocess price data.
    """
    def __init__(self, tickers, start_date, end_date, interval='1d'):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.data = None

    def fetch(self):
        """
        Download adjusted close prices and forward-fill missing values.
        """
        df = yf.download(
            self.tickers, start=self.start_date, end=self.end_date, interval=self.interval
        )['Adj Close']
        df = df.dropna(how='all').ffill().bfill()
        self.data = df
        return df

    def get_returns(self):
        """
        Compute daily returns.
        """
        if self.data is None:
            raise ValueError("Fetch data first using .fetch()")
        return self.data.pct_change().dropna()

    def get_numpy(self):
        """
        Return price array for environment.
        """
        if self.data is None:
            raise ValueError("Fetch data first using .fetch()")
        return self.data.values
