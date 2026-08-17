# Portfolio Analytics Dashboard

Interactive portfolio analytics dashboard for Eurozone blue-chip companies stocks, retrieved with `yfinance`. The application lets construct a portfolio, compare periodic rebanlancing strategy/buy an hold strategy to an Euro Stoxx 50 ETF benchmark, and analyze performance, risk, attribution within a configurable analysis time lapse. The application allows optional transaction costs at each rebalancing date based on traded notional.
The ETF tracks the performance of the Euro Stoxx 50 net return index and assumes that gross dividends are reinvested in the index.

## Features

### Performance analytics

- Total and annualized return
- Portfolio value and cumulative performance
- Annualized volatility
- Sharpe ratio
- Profit and loss
- Maximum drawdown
- Asset-level performance attribution

### Market-risk analytics

- Information ratio
- Correlation with the benchmark
- Asset correlation matrix
- Marginal and component contributions to portfolio volatility
- Return-distribution analysis

## Methodology

### Portfolio return

For a constant-weight portfolio, the return at time \(t\) is:

$$
R_{p,t} = \sum_{i=1}^{n} w_i R_{i,t}
$$

For periodic rebalancing, the engine tracks each position independently and restores the target weights only on scheduled rebalancing dates.

### Annualized volatility

Daily volatility is annualized using 252 trading days:

$$
\sigma_{annual} = \sigma_{daily}\sqrt{252}
$$

### Sharpe ratio

$$
Sharpe = \frac{E[R_p-R_f]}{\sigma(R_p-R_f)}\sqrt{252}
$$

### Portfolio volatility

Given the weight vector \(w\) and covariance matrix \(\Sigma\):

$$
\sigma_p = \sqrt{w^T\Sigma w}
$$

### Risk contribution

The component contribution of asset \(i\) to portfolio volatility is:

$$
RC_i = w_i\frac{(\Sigma w)_i}{\sigma_p}
$$

The component contributions sum to total portfolio volatility.

### Historical VaR and Expected Shortfall

Historical VaR is obtained from the lower tail of observed portfolio returns. Expected Shortfall is the average loss among observations beyond the VaR threshold. Both measures are displayed as positive losses in the dashboard.

### Transaction costs

At each rebalancing date, traded notional is calculated as:

$$
Turnover_t = \sum_{i=1}^{n}\left|V^{target}_{i,t}-V_{i,t}\right|
$$

Transaction costs are then deducted from portfolio value:

$$
Cost_t = Turnover_t \times cost\ rate
$$

## Running app

### Requirements

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Technology stack

- yfinance 
- Pandas and NumPy
- Streamlit
- Plotly
- pytest
