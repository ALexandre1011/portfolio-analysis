import numpy as np
import pandas as pd


def historical_var(returns: pd.Series, confidence_level: float) -> float:
    """Calculate historical Value at Risk expressed as a positive loss."""
    _validate_returns(returns)

    quantile = returns.quantile(1 - confidence_level)

    return max(0.0, -quantile)


def historical_expected_shortfall(returns: pd.Series, confidence_level: float) -> float:
    """Calculate historical Expected Shortfall expressed as a positive loss."""
    _validate_returns(returns)

    quantile = returns.quantile(1 - confidence_level)
    tail_returns = returns[returns <= quantile]

    if tail_returns.empty:
        return np.nan

    return max(0.0, -tail_returns.mean())


def beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate portfolio beta relative to a benchmark."""
    aligned_returns = _align_returns(
        portfolio_returns,
        benchmark_returns,
    )

    portfolio = aligned_returns["portfolio"]
    benchmark = aligned_returns["benchmark"]

    benchmark_variance = benchmark.var(ddof=1)

    if np.isclose(benchmark_variance, 0):
        return np.nan

    covariance = portfolio.cov(benchmark)

    return covariance / benchmark_variance


def tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int
) -> float:
    """Calculate annualized tracking error."""
    aligned_returns = _align_returns(
        portfolio_returns,
        benchmark_returns,
    )

    active_returns = aligned_returns["portfolio"] - aligned_returns["benchmark"]

    return active_returns.std(ddof=1) * np.sqrt(periods_per_year)


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int
) -> float:
    """Calculate the annualized information ratio."""
    aligned_returns = _align_returns(
        portfolio_returns,
        benchmark_returns,
    )

    active_returns = aligned_returns["portfolio"] - aligned_returns["benchmark"]

    active_volatility = active_returns.std(ddof=1)

    if np.isclose(active_volatility, 0):
        return np.nan

    return active_returns.mean() / active_volatility * np.sqrt(periods_per_year)


def correlation_with_benchmark(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate correlation with the benchmark."""
    aligned_returns = _align_returns(
        portfolio_returns,
        benchmark_returns,
    )

    return aligned_returns["portfolio"].corr(aligned_returns["benchmark"])


def risk_summary(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    confidence_level: float = 0.95,
    periods_per_year: int = 252,
) -> pd.Series:
    """Return the principal portfolio risk indicators."""
    return pd.Series(
        {
            "Historical VaR": historical_var(
                portfolio_returns,
                confidence_level,
            ),
            "Expected Shortfall": historical_expected_shortfall(
                portfolio_returns,
                confidence_level,
            ),
            "Beta": beta(
                portfolio_returns,
                benchmark_returns,
            ),
            "Tracking error": tracking_error(
                portfolio_returns,
                benchmark_returns,
                periods_per_year,
            ),
            "Information ratio": information_ratio(
                portfolio_returns,
                benchmark_returns,
                periods_per_year,
            ),
            "Benchmark correlation": correlation_with_benchmark(
                portfolio_returns,
                benchmark_returns,
            ),
        }
    )


def _align_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """Align portfolio and benchmark returns on common dates."""
    _validate_returns(portfolio_returns)
    _validate_returns(benchmark_returns)

    aligned_returns = pd.concat(
        [
            portfolio_returns.rename("portfolio"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned_returns) < 2:
        raise ValueError(
            "Portfolio and benchmark do not have enough common dates."
        )

    return aligned_returns


def _validate_returns(returns: pd.Series) -> None:
    if not isinstance(returns, pd.Series):
        raise TypeError("Returns must be provided as a pandas Series.")

    if returns.empty:
        raise ValueError("The return series cannot be empty.")

    if returns.isna().any():
        raise ValueError("The return series contains missing values.")

    if not np.isfinite(returns).all():
        raise ValueError("The return series contains infinite values.")