from datetime import date

import pandas as pd
import streamlit as st

from src.attribution import correlation_matrix, realized_performance_contributions, risk_contributions

from src.data import calculate_returns, download_prices

from src.performance import drawdown_series, performance_summary

from src.plots import (
    correlation_heatmap,
    cumulative_returns_chart,
    drawdown_chart,
    performance_contribution_chart,
    portfolio_values_chart,
    returns_distribution_chart,
    risk_contribution_chart,
    weights_chart,
)
from src.portfolio import Portfolio
from src.risk import risk_summary

STOCKS_BY_COUNTRY = {
    "Germany": {
        "ADS.DE": "Adidas",
        "ALV.DE": "Allianz",
        "BAS.DE": "BASF",
        "BAYN.DE": "Bayer",
        "BMW.DE": "BMW",
        "DBK.DE" : "Deutsche Bank",
        "DB1.DE": "Deutsche Börse",
        "DTE.DE": "Deutsche Telekom",
        "DHL.DE": "DHL Group",
        "IFX.DE": "Infineon Technologies",
        "MBG.DE": "Mercedez-Benz Group",
        "MUV2.DE": "Munich Re",
        "RHM.DE": "Rheinmetall",
        "SAP.DE": "SAP",
        "SIE.DE": "Siemens",
        "ENR.DE": "Siemens Energy",
        "VOW.DE": "Volkswagen Group"
    },
    "France": {
        "AI.PA": "Air Liquide",
        "AIR.PA": "Airbus",
        "CS.PA": "AXA",
        "BNP.PA": "BNP Paribas",
        "BN.PA": "Danone",
        "EL.PA": "EssilorLuxottica",
        "RMS.PA": "Hermès",
        "OR.PA": "L'Oréal",
        "MC.PA": "LVMH",
        "SAF.PA": "Safran",
        "SGO.PA": "Saint-Gobain",
        "SAN.PA": "Sanofi",
        "SU.PA": "Schneider Electric",
        "TTE.PA": "TotalEnergies",
        "DG.PA": "Vinci"
    },
    "Netherlands": {
        "ADYEN.AS": "Adyen",
        "AD.AS": "Ahold Delhaize",
        "ASML.AS": "ASML",
        "INGA.AS": "ING",
        "PRX.AS": "Prosus",
        "WKL.AS": "Wolters Kluwer"
    },
    "Italy": {
        "ENEL.MI": "Enel",
        "ENI.MI": "Eni",
        "RACE.MI": "Ferrari",
        "ISP.MI": "Intesa Sanpaolo",
        "UCG.MI": "UniCredit"
    },    
    "Spain": {
        "BBVA.MC": "BBVA",
        "SAN.MC": "Santander Group",
        "IBE.MC": "Iberdrola",
        "ITX.MC": "Inditex"
    },
    "Belgium": {
        "ABI.BR": "Anheuser-Busch InBev",
        "ARGX.BR": "Argenx"
    },
    "Finland": {
        "NDA-FI.HE": "Nordea Bank"
    }
}

STOCK_INFO = {
    ticker: {
        "name": company,
        "country": country,
    }
    for country, stocks in STOCKS_BY_COUNTRY.items()
    for ticker, company in stocks.items()
}

BENCHMARK_TICKER = "C50.PA"
BENCHMARK_NAME = "Amundi Core EURO STOXX 50 UCITS ETF Acc"

STRATEGIES = {
    "Daily rebalanced": 252,
    "Weekly rebalanced": 52,
    "Monthly rebalanced": 12,
    "Quarterly rebalanced": 4,
    "Annual rebalanced": 1,
    "Buy and hold": None
}

@st.cache_data(ttl=3600)
def load_market_data(
    tickers: tuple[str, ...],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Download and cache historical market prices."""
    return download_prices(tickers=list(tickers), start_date=start_date, end_date=end_date)

def reset_equal_weights(tickers: tuple[str, ...]) -> None:
    """Reset selected assets to equal weights."""
    if not tickers:
        return

    equal_weight = 100 / len(tickers)

    for ticker in tickers:
        st.session_state[f"weight_{ticker}"] = equal_weight

st.set_page_config(page_title="Portfolio analytics dashboard", layout="wide")
st.title("Portfolio analytics dashboard")
st.markdown(
    """
    This dashboard aims to build and analyze a portfolio of Eurozone blue-chip companies stocks. 
    
    It allows to compare rebalancing strategies against the Euro Stoxx 50 tracking ETF.

    Use the sidebar to configure the portfolio and assumptions.
    """
)

with st.sidebar:
    selected_countries = st.multiselect(
        label="Countries",
        options=list(STOCKS_BY_COUNTRY),
        default=list(STOCKS_BY_COUNTRY),
        placeholder="Select countries"
    )

    if not selected_countries:
        st.error("Enter at least one country.")
        st.stop()

    available_tickers = [
        ticker
        for country in selected_countries
        for ticker in STOCKS_BY_COUNTRY[country]
    ]

    tickers = st.multiselect(
        label="Stocks",
        options=available_tickers,
        default=available_tickers[:2],
        format_func=lambda ticker: f"{STOCK_INFO[ticker]['name']}",
        placeholder="Select stocks",
        max_selections=10
    )

    if not tickers:
        st.error("Enter at least one stock.")
        st.stop()

    st.text_input(
        label="Benchmark",
        value=f"{BENCHMARK_NAME}",
        disabled=True,
        help=(
            "Fixed accumulating ETF benchmark tracking the Euro Stoxx 50 Net Return index, "
            "where dividends are reinvested in the fund and managment fees are included."
        )
    )

    left, right = st.columns(2)
    with right:
        end_date = st.date_input(
            label="End date", 
            value=pd.Timestamp.today().date()
        )

    with left:
        start_date = st.date_input(
            label="Start date", 
            value=(pd.Timestamp.today() - pd.DateOffset(years=3)).date()
        )

    if start_date >= end_date:
        st.error("The end date must be later than the start date.")
        st.stop()
   
    time = (end_date - start_date).days / 365.25

    available_strategy = {
        name: frequency
        for name, frequency in STRATEGIES.items()
        if frequency is None or frequency * time >= 1
    }
    
    strategy = st.selectbox(
        label="Strategy",
        options=list(available_strategy)
    )
    
    frequency = available_strategy[strategy]

    initial_value = st.number_input(
        label="Initial portfolio value",
        min_value=1000.00,
        value=100000.00,
        step=5000.00,
        format="%.2f"
    )

    transaction_cost_bps = st.number_input(
        label="Transaction costs (bps)",
        min_value=0.00,
        max_value=100.00,
        value=5.00,
        step=1.00,
        help="Cost applied to the notional at each rebalancing trade.",
        disabled=strategy == "Buy and hold"
    )

    transaction_cost_rate = transaction_cost_bps / 10000

    if strategy == "Buy and hold":
        transaction_cost_rate = 0.0

    risk_free_rate = st.slider(
        label="Annual risk-free rate", 
        min_value=0.00, 
        max_value=0.20, 
        value=0.05, 
        step=0.001,
        format="percent"
    )

    confidence_level = st.selectbox(
        "VaR confidence level",
        options=[0.95, 0.99],
        format_func=lambda value: f"{value:.0%}"
    )

current_selection = tuple(tickers)
previous_selection = st.session_state.get("previous_ticker_selection")

# Reset weights only when the asset selection changes
if current_selection != previous_selection:
    reset_equal_weights(current_selection)

    # Remove weights belonging to deselected assets
    for key in list(st.session_state.keys()):
        if key.startswith("weight_"):
            ticker = key.removeprefix("weight_")

            if ticker not in tickers:
                del st.session_state[key]

    st.session_state["previous_ticker_selection"] = current_selection


with st.sidebar:
    st.divider()
    entered_weights = {}

    for ticker in tickers:
        weight_key = f"weight_{ticker}"

        entered_weights[ticker] = st.number_input(
            label=f"{STOCK_INFO[ticker]['name']} weight (%)",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            format="%.2f",
            key=weight_key
        )

    total_weight = sum(entered_weights.values())
    left, right = st.columns([2, 1])
    left.metric(label="Total weight", value=f"{total_weight:.2f}%")

    right.button(
        label="Reset to equal weights",
        on_click=reset_equal_weights,
        args=(current_selection,),
        type="primary"
    )

if abs(total_weight - 100) > 0.01:
    st.error("Portfolio weights sum must be 100%.")
    st.stop()

weights = {
    ticker: weight / total_weight
    for ticker, weight in entered_weights.items()
}

all_tickers = tuple(tickers + [BENCHMARK_TICKER])

try:
    prices = load_market_data(
        tickers=all_tickers,
        start_date=start_date,
        end_date=end_date,
    )

    returns = calculate_returns(prices)

except Exception as error:
    st.error(f"Unable to load market data: {error}")
    st.stop()


asset_returns = returns[tickers]
benchmark_returns = returns[BENCHMARK_TICKER]

portfolio = Portfolio(
    returns=asset_returns,
    weights=weights,
    initial_value=initial_value
)

asset_values_history, transaction_costs, turnovers = (
    portfolio.simulate(
        frequency=frequency,
        transaction_cost_rate=transaction_cost_rate,
    )
)

selected_values = asset_values_history.sum(axis=1)
selected_values.name = strategy

previous_values = selected_values.shift(1)
previous_values.iloc[0] = initial_value

selected_returns = selected_values / previous_values - 1

selected_returns.name = strategy

weights_history = asset_values_history.div(selected_values, axis="index")

allocation_weights = weights_history.iloc[-1]

total_transaction_costs = transaction_costs.sum()

investment_years = (turnovers.index[-1] - turnovers.index[0]).days / 365.25

annualized_turnover = (
    float(turnovers.sum()) / investment_years
    if investment_years > 0
    else 0.0
)

performance_attribution = (
    realized_performance_contributions(
        asset_values=asset_values_history,
        returns=asset_returns,
        initial_value=initial_value,
    )
)

if total_transaction_costs > 0:
    performance_attribution.loc["Transaction costs"] = (-total_transaction_costs / initial_value)

benchmark_values = (initial_value * (1 + benchmark_returns).cumprod())

comparison_values = pd.concat(
    [selected_values.rename(strategy), benchmark_values.rename(BENCHMARK_TICKER)],
    axis=1,
    join="inner"
)

comparison_returns = pd.concat(
    [selected_returns.rename(strategy), benchmark_returns.rename(BENCHMARK_TICKER)],
    axis=1,
    join="inner"
)

try: 
    performance_metrics = performance_summary(
        returns=selected_returns,
        portfolio_returns=selected_returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate
    )

    risk_metrics = risk_summary(
        portfolio_returns=selected_returns,
        benchmark_returns=benchmark_returns,
        confidence_level=confidence_level
    )

except Exception as error:
    st.error(f"Unable to compute performance and risk metrics: {error}")
    st.stop()

portfolio_drawdowns = drawdown_series(selected_returns)

risk_attribution = risk_contributions(returns=asset_returns, weights=allocation_weights)

correlations = correlation_matrix(asset_returns)


current_value = selected_values.iloc[-1]
pnl = current_value - initial_value

var_rate, expected_shortfall_rate = (
    risk_metrics["Historical VaR"], risk_metrics["Expected Shortfall"]
)

var_amount, expected_shortfall_amount = (
    current_value * var_rate, current_value * expected_shortfall_rate
)

overview_tab, performance_tab, risk_tab, holdings_tab = st.tabs(
    [
        "Overview",
        "Performance",
        "Risk",
        "Holdings"
    ]
)

with overview_tab:
    first_row = st.columns(3)

    first_row[0].metric(
        label="Portfolio value",
        value=f"{current_value:,.2f} €",
        delta=f"{performance_metrics["Total return"]:.2%}"
    )

    first_row[1].metric(
        label="PnL",
        value=f"{pnl:,.2f} €"
    )

    first_row[2].metric(
        label="Annualized return",
        value=f"{performance_metrics["Annualized return"]:.2%}"
    )

    second_row = st.columns(3)

    second_row[0].metric(
        label="Annualized volatility",
        value=f"{performance_metrics["Annualized volatility"]:.2%}"
    )

    second_row[1].metric(
        label="Annualized turnover",
        value=f"{annualized_turnover:.2%}"
    )

    second_row[2].metric(
        label="Total transaction costs",
        value=f"{total_transaction_costs:,.2f} €"
    )

    st.plotly_chart(
        portfolio_values_chart(comparison_values),
        width="stretch"
    )

    st.plotly_chart(
        weights_chart(allocation_weights),
        width="stretch"
    )

with performance_tab:
    first_row = st.columns(3)

    first_row[0].metric(
        label="Tracking error",
        value=f"{risk_metrics["Tracking error"]:.2%}"
    )

    first_row[1].metric(
        label="Maximum drawdown",
        value=f"{performance_metrics["Maximum drawdown"]:.2%}"
    )

    first_row[2].metric(
        label="Beta",
        value=f"{risk_metrics["Beta"]:.2f}"
    )

    second_row = st.columns(3)

    second_row[0].metric(
        label="Annualized active return",
        value=f"{performance_metrics["Active return"]:.2%}"
    )

    second_row[1].metric(
        label="Sharpe ratio",
        value=f"{performance_metrics['Sharpe ratio']:.2f}"
    )

    second_row[2].metric(
        label="Information ratio",
        value=f"{risk_metrics['Information ratio']:.2f}"
    )

    st.plotly_chart(
        cumulative_returns_chart(comparison_returns),
        width="stretch",
        key="cumulative_returns_chart"
    )

    left, right = st.columns(2)

    with right:
        st.plotly_chart(
            drawdown_chart(portfolio_drawdowns),
            width="stretch",
            key="drawdown_chart"
        )

    with left:
        st.plotly_chart(
            performance_contribution_chart(performance_attribution),
            width="stretch",
            key="performance_contribution_chart"
        )

with risk_tab:
    first_row = st.columns(3)

    first_row[0].metric(
        label=f"Daily VaR {confidence_level:.0%}",
        value=f"-{var_amount:,.2f} €",
        delta=f"-{var_rate:.2%}"
    )

    first_row[1].metric(
        label="Expected Shortfall",
        value=f"-{expected_shortfall_amount:,.2f} €",
        delta=f"-{expected_shortfall_rate:.2%}"
    )

    first_row[2].metric(
        label="Benchmark correlation",
        value=f"{risk_metrics["Benchmark correlation"]:.2f}"
    )

    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            returns_distribution_chart(
                selected_returns,
                var_value=var_rate,
                expected_shortfall=expected_shortfall_rate,
            ),
            width="stretch",
            key="returns_distribution_chart"
        )

    with right:
        st.plotly_chart(
            risk_contribution_chart(risk_attribution),
            width="stretch",
            key="risk_contribution_chart"
        )

with holdings_tab:
    asset_performance = (1 + asset_returns).prod() - 1

    holdings_table = pd.DataFrame(
        {
            "Company": [
                STOCK_INFO[ticker]["name"]
                for ticker in tickers
            ],
            "Country": [
                STOCK_INFO[ticker]["country"]
                for ticker in tickers
            ],
            "Target weight": pd.Series(weights),
            "Current weight": allocation_weights,
            "Asset performance": asset_performance,
            "Risk contribution": risk_attribution["Risk contribution %"],
        }
    )

    st.dataframe(
        holdings_table.style.format(
            {
                "Target weight": "{:.2%}",
                "Current weight": "{:.2%}",
                "Asset performance": "{:.2%}",
                "Risk contribution": "{:.2%}",
            }
        ),
        width="stretch",
        hide_index=True
    )

    st.plotly_chart(
        correlation_heatmap(correlations),
        width="stretch",
        key="correlation_heatmap"
    )