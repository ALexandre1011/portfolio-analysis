from typing import Literal, TypeAlias

import numpy as np
import pandas as pd


RebalancingFrequency: TypeAlias = Literal[1, 4, 12, 52, 252, None]

class Portfolio:

    VALID_FREQUENCIES = {1, 4, 12, 52, 252, None}
    def __init__(
        self,
        returns: pd.DataFrame,
        weights: dict[str, float],
        initial_value: float = 100000
    ):
        self.returns = returns.copy()
        self.weights = pd.Series(weights, dtype=float)
        self.initial_value = float(initial_value)

        self._validate_inputs()
        self.weights = self.weights.reindex(self.returns.columns)

    def _validate_inputs(self) -> None:
        """Validate portfolio returns, weights and initial value."""

        if self.returns.empty:
            raise ValueError("The returns DataFrame cannot be empty.")

        if not isinstance(self.returns.index, pd.DatetimeIndex):
            raise TypeError("Returns must use a DatetimeIndex.")

        if self.returns.index.has_duplicates:
            raise ValueError("Return dates must be unique.")

        if not self.returns.index.is_monotonic_increasing:
            raise ValueError("Return dates must be sorted.")

        if self.returns.columns.duplicated().any():
            raise ValueError("Asset names must be unique.")

        if not all(
            pd.api.types.is_numeric_dtype(dtype)
            for dtype in self.returns.dtypes
        ):
            raise TypeError("Returns must contain only numeric values.")

        returns_array = self.returns.to_numpy(dtype=float)

        if not np.isfinite(returns_array).all():
            raise ValueError("Returns cannot contain NaN or infinite values.")

        if (returns_array < -1.0).any():
            raise ValueError("Simple returns cannot be lower than -100%.")

        if self.weights.empty:
            raise ValueError(
                "The portfolio must contain at least one asset."
            )

        return_assets = set(self.returns.columns)
        weight_assets = set(self.weights.index)

        missing_weights = return_assets - weight_assets
        unexpected_weights = weight_assets - return_assets

        if missing_weights:
            raise ValueError(
                "Missing weights for: "
                + ", ".join(sorted(missing_weights))
            )

        if unexpected_weights:
            raise ValueError(
                "Unexpected weights for: "
                + ", ".join(sorted(unexpected_weights))
            )

        if not np.isfinite(self.weights.to_numpy()).all():
            raise ValueError(
                "Weights cannot contain NaN or infinite values."
            )

        if (self.weights < 0).any():
            raise ValueError("Negative weights are not allowed.")

        if not np.isclose(
            float(self.weights.sum()),
            1.0,
            atol=1e-8,
        ):
            raise ValueError("Portfolio weights must sum to 1.")

        if self.initial_value <= 0:
            raise ValueError("Initial portfolio value must be positive.")

    def _rebalance(
        self,
        asset_values: pd.Series,
        transaction_cost_rate: float,
    ) -> tuple[pd.Series, float, float]:
        """Rebalance asset values and calculate costs and gross turnover."""

        portfolio_value = float(asset_values.sum())

        if portfolio_value <= 0:
            raise ValueError("Portfolio value must be positive.")

        current_weights = asset_values / portfolio_value

        gross_turnover = (current_weights - self.weights).abs().sum()

        traded_notional = portfolio_value * gross_turnover

        transaction_cost = traded_notional * transaction_cost_rate

        value_after_costs = portfolio_value - transaction_cost

        rebalanced_values = value_after_costs * self.weights

        return rebalanced_values, transaction_cost, gross_turnover

    def simulate(
        self,
        frequency: RebalancingFrequency,
        transaction_cost_rate: float = 0.0,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Simulate portfolio values, costs and gross turnover."""

        if not 0.0 <= transaction_cost_rate < 1.0:
            raise ValueError("Transaction cost rate must be between 0 and 1.")

        asset_values = (self.initial_value * self.weights).copy()

        values_history: list[pd.Series] = []
        transaction_cost_history: list[float] = []
        turnover_history: list[float] = []

        previous_date: pd.Timestamp | None = None

        for index_value, period_returns in self.returns.iterrows():
            current_date = pd.Timestamp(index_value)

            transaction_cost = 0.0
            gross_turnover = 0.0

            if (
                previous_date is not None
                and self._should_rebalance(
                    previous_date=previous_date,
                    current_date=current_date,
                    frequency=frequency,
                )
            ):
                (
                    asset_values,
                    transaction_cost,
                    gross_turnover,
                ) = self._rebalance(
                    asset_values=asset_values,
                    transaction_cost_rate=transaction_cost_rate,
                )

            asset_values = asset_values * (
                1.0 + period_returns
            )

            values_history.append(asset_values.copy())
            transaction_cost_history.append(transaction_cost)
            turnover_history.append(gross_turnover)

            previous_date = current_date

        values = pd.DataFrame(
            values_history,
            index=self.returns.index,
            columns=self.returns.columns,
        )

        transaction_costs = pd.Series(
            transaction_cost_history,
            index=self.returns.index,
            dtype=float,
            name="Transaction costs",
        )

        turnovers = pd.Series(
            turnover_history,
            index=self.returns.index,
            dtype=float,
            name="Gross turnover",
        )

        return values, transaction_costs, turnovers
    
    def portfolio_returns(self) -> pd.Series:
        """Calculate portfolio returns using constant weights."""
        return self.returns.mul(self.weights, axis="columns").sum(axis=1)

    def cumulative_returns(self) -> pd.Series:
        """Calculate the cumulative performance of the portfolio."""
        returns = self.portfolio_returns()

        return (1 + returns).cumprod() - 1

    @staticmethod
    def _should_rebalance(
        previous_date: pd.Timestamp,
        current_date: pd.Timestamp,
        frequency: RebalancingFrequency,
    ) -> bool:
        """Determine whether rebalancing occurs before a return."""

        if frequency is None:
            return False

        if frequency == 252:
            return True

        if frequency == 52:
            return (
                previous_date.to_period("W")
                != current_date.to_period("W")
            )

        if frequency == 12:
            return (
                previous_date.year,
                previous_date.month,
            ) != (
                current_date.year,
                current_date.month,
            )

        if frequency == 4:
            return (
                previous_date.year,
                previous_date.quarter,
            ) != (
                current_date.year,
                current_date.quarter,
            )

        if frequency == 1:
            return previous_date.year != current_date.year

        return False
    
    def asset_values_history(
        self,
        frequency: RebalancingFrequency,
        transaction_cost_rate: float = 0.0
    ) -> pd.DataFrame:
        asset_values, _, _ = self.simulate(frequency, transaction_cost_rate)

        return asset_values


    def strategy_values(
        self,
        frequency: RebalancingFrequency,
        transaction_cost_rate: float = 0.0,
    ) -> pd.Series:
        asset_values, _, _ = self.simulate(frequency, transaction_cost_rate)

        values = asset_values.sum(axis=1)
        values.name = "Portfolio value"

        return values


    def strategy_returns(
        self,
        frequency: RebalancingFrequency,
        transaction_cost_rate: float = 0.0
    ) -> pd.Series:
        values = self.strategy_values(frequency, transaction_cost_rate)

        previous_values = values.shift(1)
        previous_values.iloc[0] = self.initial_value

        returns = values / previous_values - 1
        returns.name = "Portfolio returns"

        return returns


    def strategy_weights(
        self,
        frequency: RebalancingFrequency,
        transaction_cost_rate: float = 0.0
    ) -> pd.DataFrame:
        asset_values = self.asset_values_history(frequency, transaction_cost_rate)

        return asset_values.div(asset_values.sum(axis=1), axis="index")

    def daily_rebalanced_values(self) -> pd.Series:
        return self.strategy_values(252)

    def daily_rebalanced_returns(self) -> pd.Series:
        return self.strategy_returns(252)

    def buy_and_hold_values(self) -> pd.Series:
        return self.strategy_values(None)

    def buy_and_hold_returns(self) -> pd.Series:
        return self.strategy_returns(None)

    def buy_and_hold_cumulative_returns(self) -> pd.Series:
        """Calculate cumulative returns of the buy-and-hold portfolio."""
        values = self.buy_and_hold_values()

        cumulative_returns = values / self.initial_value - 1
        cumulative_returns.name = "Buy-and-hold cumulative returns"

        return cumulative_returns

    def current_weights(self) -> pd.DataFrame:
        return self.strategy_weights(None)

    def strategy_transaction_costs(
        self,
        frequency: RebalancingFrequency,
        transaction_cost_rate: float = 0.0,
    ) -> pd.Series:
        _, costs, _ = self.simulate(frequency,  transaction_cost_rate)

        return costs


    def strategy_turnover(
        self,
        frequency: RebalancingFrequency,
        transaction_cost_rate: float = 0.0,
    ) -> pd.Series:
        _, _, turnovers = self.simulate(frequency, transaction_cost_rate)

        return turnovers