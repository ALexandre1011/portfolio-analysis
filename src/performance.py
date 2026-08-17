import numpy as np
import pandas as pd


def total_return(returns: pd.Series) -> float:
    """Calculate the total compounded return."""
    _validate_returns(returns)

    return float((1 + returns).prod()) - 1


def annualized_return(returns: pd.Series, periods_per_year: int) -> float:
    """Calculate the annualized compounded return."""
    _validate_returns(returns)

    number_of_periods = len(returns)
    cumulative_growth = (1 + returns).prod()

    return cumulative_growth ** (periods_per_year / number_of_periods) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int) -> float:
    """Calculate annualized volatility."""
    _validate_returns(returns)

    return returns.std(ddof=1) * np.sqrt(periods_per_year)

def active_return(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int
) -> float:
    """Calculate annualized portfolio return minus benchmark return."""

    aligned_returns = pd.concat(
        [
            portfolio_returns.rename("portfolio"),
            benchmark_returns.rename("benchmark"),
        ], axis=1, join="inner").dropna()

    if aligned_returns.empty:
        raise ValueError("Portfolio and benchmark returns have no common observations.")

    portfolio_return = annualized_return(
        aligned_returns["portfolio"],
        periods_per_year=periods_per_year,
    )

    benchmark_return = annualized_return(
        aligned_returns["benchmark"],
        periods_per_year=periods_per_year,
    )

    return float(portfolio_return - benchmark_return)

def sharpe_ratio(
    returns: pd.Series, 
    risk_free_rate: float, 
    periods_per_year: int
) -> float:
    """Calculate the annualized Sharpe ratio."""
    _validate_returns(returns)

    periodic_risk_free_rate = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess_returns = returns - periodic_risk_free_rate
    volatility = excess_returns.std(ddof=1)

    if np.isclose(volatility, 0):
        return np.nan

    return excess_returns.mean() / volatility * np.sqrt(periods_per_year)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Calculate the drawdown through time."""
    _validate_returns(returns)

    wealth_index = (1 + returns).cumprod()

    previous_peaks = wealth_index.cummax().clip(lower=1)

    drawdowns = wealth_index / previous_peaks - 1
    drawdowns.name = "Drawdown"

    return drawdowns


def maximum_drawdown(returns: pd.Series) -> float:
    """Calculate the maximum observed drawdown."""
    return drawdown_series(returns).min()


def performance_summary(
    returns: pd.Series, 
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float, 
    periods_per_year: int = 252
) -> pd.Series:
    """Return the principal portfolio performance indicators."""
    return pd.Series(
        {
            "Total return": total_return(returns),
            "Annualized return": annualized_return(returns, periods_per_year),
            "Annualized volatility": annualized_volatility(returns, periods_per_year),
            "Active return": active_return(portfolio_returns, benchmark_returns, periods_per_year),
            "Sharpe ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
            "Maximum drawdown": maximum_drawdown(returns)
        }
    )


def _validate_returns(returns: pd.Series) -> None:
    """Validate a return series before calculation."""
    if not isinstance(returns, pd.Series):
        raise TypeError("Returns must be provided as a pandas Series.")

    if returns.empty:
        raise ValueError("The return series cannot be empty.")

    if returns.isna().any():
        raise ValueError("The return series contains missing values.")

    if not np.isfinite(returns).all():
        raise ValueError("The return series contains infinite values.")

    if (returns <= -1).any():
        raise ValueError("Returns cannot be lower than or equal to -100%.")