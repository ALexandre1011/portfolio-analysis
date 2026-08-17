import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


POSITIVE_COLOR = "#2E8B57"
NEGATIVE_COLOR = "#C0392B"
PRIMARY_COLOR = "#1F77B4"
SECONDARY_COLOR = "#FF7F0E"


def portfolio_values_chart(values: pd.DataFrame) -> go.Figure:
    """Plot one or several portfolio value series."""
    if values.empty:
        raise ValueError("Portfolio values cannot be empty.")

    fig = go.Figure()

    for column in values.columns:
        fig.add_trace(
            go.Scatter(
                x=values.index,
                y=values[column],
                mode="lines",
                name=str(column),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    "%{y:,.2f}<extra>%{fullData.name}</extra>"
                ),
            )
        )

    return _apply_layout(
        fig,
        title="Portfolio value",
        yaxis_title="Value",
    )


def cumulative_returns_chart(returns: pd.DataFrame) -> go.Figure:
    """Plot cumulative returns for several strategies."""
    if returns.empty:
        raise ValueError("Returns cannot be empty.")

    cumulative_returns = (1 + returns).cumprod() - 1

    fig = go.Figure()

    for column in cumulative_returns.columns:
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[column],
                mode="lines",
                name=str(column),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    "%{y:.2%}<extra>%{fullData.name}</extra>"
                ),
            )
        )

    fig = _apply_layout(
        fig,
        title="Cumulative performance",
        yaxis_title="Cumulative return",
    )

    fig.update_yaxes(tickformat=".0%")

    return fig


def drawdown_chart(drawdowns: pd.Series) -> go.Figure:
    """Plot portfolio drawdowns."""
    if drawdowns.empty:
        raise ValueError("Drawdowns cannot be empty.")

    fig = go.Figure(
        go.Scatter(
            x=drawdowns.index,
            y=drawdowns,
            mode="lines",
            name="Drawdown",
            line={"color": NEGATIVE_COLOR},
            fill="tozeroy",
            fillcolor="rgba(192, 57, 43, 0.25)",
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "%{y:.2%}<extra></extra>"
            ),
        )
    )

    fig = _apply_layout(
        fig,
        title="Portfolio drawdown",
        yaxis_title="Drawdown",
    )

    fig.update_yaxes(tickformat=".0%")

    return fig


def weights_chart(weights: pd.Series | dict[str, float]) -> go.Figure:
    """Plot portfolio weights as a donut chart."""
    weights = pd.Series(weights, dtype=float)

    data = weights.rename("Weight").reset_index()
    data.columns = ["Asset", "Weight"]

    fig = px.pie(
        data,
        names="Asset",
        values="Weight",
        title="Portfolio allocation",
    )

    fig.update_traces(
        textposition="inside",
        texttemplate="%{label}<br>%{percent}",
        hovertemplate=(
            "%{label}<br>"
            "Weight: %{value:.2%}<extra></extra>"
        ),
    )

    fig.update_layout(
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend_title_text="Asset",
    )

    return fig


def performance_contribution_chart(contributions: pd.Series) -> go.Figure:
    """Plot asset contributions to total portfolio performance."""
    if contributions.empty:
        raise ValueError("Performance contributions cannot be empty.")

    colors = [POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR for value in contributions]

    fig = go.Figure(
        go.Bar(
            x=contributions.index,
            y=contributions.values,
            marker_color=colors,
            text=contributions.values,
            texttemplate="%{text:.2%}",
            textposition="outside",
            hovertemplate=(
                "%{x}<br>"
                "Contribution: %{y:.2%}<extra></extra>"
            ),
        )
    )

    fig = _apply_layout(
        fig,
        title="Contribution to total performance",
        yaxis_title="Performance contribution",
    )

    fig.update_yaxes(tickformat=".0%")

    return fig


def risk_contribution_chart(risk_attribution: pd.DataFrame) -> go.Figure:
    """Plot percentage contributions to portfolio volatility."""
    column = "Risk contribution %"

    if column not in risk_attribution.columns:
        raise ValueError(f"Risk attribution must contain '{column}'.")

    contributions = risk_attribution[column]

    colors = [POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR for value in contributions]

    fig = go.Figure(
        go.Bar(
            x=contributions.index,
            y=contributions.values,
            marker_color=colors,
            text=contributions.values,
            texttemplate="%{text:.2%}",
            textposition="outside",
            hovertemplate=(
                "%{x}<br>"
                "Risk contribution: %{y:.2%}<extra></extra>"
            ),
        )
    )

    fig = _apply_layout(
        fig,
        title="Contribution to portfolio risk",
        yaxis_title="Share of total volatility",
    )

    fig.update_yaxes(tickformat=".0%")

    return fig


def correlation_heatmap(correlations: pd.DataFrame) -> go.Figure:
    """Plot the asset return correlation matrix."""
    if correlations.empty:
        raise ValueError("Correlation matrix cannot be empty.")

    fig = go.Figure(
        go.Heatmap(
            z=correlations.values,
            x=correlations.columns,
            y=correlations.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            text=correlations.round(2).values,
            texttemplate="%{text:.2f}",
            hovertemplate=(
                "%{y} / %{x}<br>"
                "Correlation: %{z:.2f}<extra></extra>"
            ),
            colorbar={"title": "Correlation"},
        )
    )

    fig.update_layout(
        title="Asset correlation matrix",
        template="plotly_white",
        margin={"l": 30, "r": 30, "t": 60, "b": 30},
        xaxis={"side": "bottom"},
    )

    return fig


def returns_distribution_chart(
    returns: pd.Series,
    var_value: float | None = None,
    expected_shortfall: float | None = None
) -> go.Figure:
    """Plot the return distribution with optional VaR and ES thresholds."""
    if returns.empty:
        raise ValueError("Returns cannot be empty.")

    fig = go.Figure(
        go.Histogram(
            x=returns,
            nbinsx=50,
            histnorm="probability density",
            name="Returns",
            marker_color=PRIMARY_COLOR,
            opacity=0.75,
            hovertemplate=(
                "Return: %{x:.2%}<br>"
                "Density: %{y:.2f}<extra></extra>"
            ),
        )
    )

    if var_value is not None:
        fig.add_vline(
            x=-var_value,
            line_color=SECONDARY_COLOR,
            line_dash="dash",
            annotation_text="VaR",
            annotation_position="top",
        )

    if expected_shortfall is not None:
        fig.add_vline(
            x=-expected_shortfall,
            line_color=NEGATIVE_COLOR,
            line_dash="dash",
            annotation_text="Expected Shortfall",
            annotation_position="top left",
        )

    fig = _apply_layout(
        fig,
        title="Distribution of portfolio returns",
        xaxis_title="Daily return",
        yaxis_title="Density",
    )

    fig.update_xaxes(tickformat=".1%")

    return fig


def _apply_layout(
    fig: go.Figure,
    title: str,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
) -> go.Figure:
    """Apply a consistent layout to Plotly figures."""
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode="x unified",
        legend_title_text="",
        margin={"l": 30, "r": 30, "t": 60, "b": 30},
    )

    return fig