from datetime import date

import pandas as pd
import yfinance as yf


def download_prices(
    tickers: list[str], 
    start_date: str | date, 
    end_date: str | date
) -> pd.DataFrame:
    """Download closing prices for several financial assets."""
    if not tickers:
        raise ValueError("The ticker list cannot be empty.")

    tickers = [ticker.strip().upper() for ticker in tickers]

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError("No market data was downloaded.")

    prices = data["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.sort_index()
    prices = prices.dropna(how="all")

    missing_tickers = [
        ticker for ticker in tickers
        if ticker not in prices.columns
        or prices[ticker].dropna().empty
    ]

    if missing_tickers:
        raise ValueError(f"No price data found for: {', '.join(missing_tickers)}")
    prices = prices.dropna(how="any")

    if prices.empty:
        raise ValueError("No common price dates are available for all selected assets.")

    return prices

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple returns from historical prices."""
    if prices.empty:
        raise ValueError("The price DataFrame cannot be empty.")

    if prices.isna().any().any():
        raise ValueError("The price DataFrame contains missing values.")

    returns = prices.pct_change(fill_method=None).dropna(how="any")

    if returns.empty:
        raise ValueError("At least two price observations are required.")

    return returns