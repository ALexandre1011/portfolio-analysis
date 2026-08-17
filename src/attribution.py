import numpy as np
import pandas as pd


def performance_contributions(
    returns: pd.DataFrame,
    weights: pd.Series | dict[str, float]
) -> pd.Series:
    """Calculate each asset's exact contribution to the total return."""
    returns, weights = _prepare_inputs(returns, weights)

    portfolio_returns = returns.mul(weights, axis="columns").sum(axis=1)

    portfolio_wealth = (1 + portfolio_returns).cumprod()

    previous_wealth = portfolio_wealth.shift(1, fill_value=1.0)

    periodic_contributions = returns.mul(weights, axis="columns").mul(
        previous_wealth,
        axis="index",
    )

    contributions = periodic_contributions.sum()
    contributions.name = "Performance contribution"

    return contributions

def realized_performance_contributions(
    asset_values: pd.DataFrame,
    returns: pd.DataFrame,
    initial_value: float
) -> pd.Series:
    """Calculate exact asset contributions to portfolio return."""
    if initial_value <= 0:
        raise ValueError("Initial portfolio value must be positive.")

    asset_values, returns = asset_values.align(returns, join="inner", axis=0)

    asset_values, returns = asset_values.align( returns, join="inner", axis=1)

    values_before_returns = asset_values.div(1 + returns)

    periodic_pnl = asset_values - values_before_returns

    contributions = periodic_pnl.sum() / initial_value

    contributions.name = "Performance contribution"

    return contributions

def cumulative_performance_contributions(
    returns: pd.DataFrame,
    weights: pd.Series | dict[str, float]
) -> pd.DataFrame:
    """
    Calculate cumulative performance contributions through time.

    At each date, the sum across assets equals the portfolio's
    cumulative return.
    """
    returns, weights = _prepare_inputs(returns, weights)

    portfolio_returns = returns.mul(weights, axis="columns").sum(axis=1)

    portfolio_wealth = (1 + portfolio_returns).cumprod()
    previous_wealth = portfolio_wealth.shift(1, fill_value=1.0)

    periodic_contributions = returns.mul(weights, axis="columns").mul(
        previous_wealth,
        axis="index",
    )

    return periodic_contributions.cumsum()


def risk_contributions(
    returns: pd.DataFrame,
    weights: pd.Series | dict[str, float],
    periods_per_year: int = 252
) -> pd.DataFrame:
    """
    Calculate component contributions to annualized volatility.

    Component contributions sum to total portfolio volatility.
    """
    returns, weights = _prepare_inputs(returns, weights)

    covariance_matrix = returns.cov() * periods_per_year

    portfolio_variance = weights.to_numpy() @ covariance_matrix.to_numpy() @ weights.to_numpy()

    portfolio_volatility = np.sqrt(portfolio_variance)

    if np.isclose(portfolio_volatility, 0):
        raise ValueError("Portfolio volatility is equal to zero.")

    marginal_contributions = (covariance_matrix @ weights) / portfolio_volatility

    component_contributions = weights * marginal_contributions

    percentage_contributions = component_contributions / portfolio_volatility

    return pd.DataFrame(
        {
            "Weight": weights,
            "Marginal contribution": marginal_contributions,
            "Risk contribution": component_contributions,
            "Risk contribution %": percentage_contributions,
        }
    )


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate the asset return correlation matrix."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("Returns must be a pandas DataFrame.")

    if returns.empty:
        raise ValueError("The returns DataFrame cannot be empty.")

    return returns.corr()


def _prepare_inputs(
    returns: pd.DataFrame,
    weights: pd.Series | dict[str, float]
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate and align returns and portfolio weights."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("Returns must be a pandas DataFrame.")

    if returns.empty:
        raise ValueError("The returns DataFrame cannot be empty.")

    if returns.isna().any().any():
        raise ValueError("Returns contain missing values.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns contain infinite values.")

    weights = pd.Series(weights, dtype=float)

    missing_assets = set(returns.columns) - set(weights.index)
    extra_assets = set(weights.index) - set(returns.columns)

    if missing_assets:
        raise ValueError(
            f"Missing weights for: {', '.join(sorted(missing_assets))}"
        )

    if extra_assets:
        raise ValueError(
            f"Unknown assets in weights: "
            f"{', '.join(sorted(extra_assets))}"
        )

    weights = weights.reindex(returns.columns)

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")
    return returns, weights