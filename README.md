# ARIMA Model Predictive Forecasting Analysis Strategy

## Overview

This project implements and backtests an **ARIMA Time Series Model to forecast stock prices** using Python. The strategy generates a **buy signal** when:

1. The **prediction for the closing price of the next day is higher than the previous**, and

---

## How The Strategy Works

1. **Calculate ARIMA parameters p, d, and q** for the chosen stock and time frame.
2. **Calculate the prediction for the next day** to determine its trend direction.
3. **Generate trading signals**:
   - **Long position (buy)** if:
     - Closing price prediction of the next day is greater than the prediction of the previous day
   - **Hold cash (no position)** otherwise.
4. **Backtest the strategy** to evaluate its cumulative returns versus a buy-and-hold benchmark.

---

## Strategy Motivation

1. **The ARIMA model** can be a powerful tool is predicting future movements.
2. Using the model **could generate an accurate prediction** of how the stock is expected to move on the next day.
3. Hence, the model forces a buy signal  if the model predicts **a price increase on the following day**.
4. This strategy, in theory, works best in **non-volatile markets where movement is smooth and predictable**, though this strategy may still be applied in other markets.

---

## Technologies Used

- **Python 3**
- `pandas` for data manipulation
- `yfinance` for downloading historical stock data
- `matplotlib` for data visualisation
- `numpy` for numerical calculations
- `datetime` for creating time frames
- `statsmodels` for implementing the technical models
- `scikit-learn` for calculating error margins

---

## Results

The backtest plots:

- **Historic prices** of the stock provided.
- **Model predictions** on the test data.
- **Cumulative returns** of the predictive model strategy.
- **Cumulative returns** of the buy-and-hold strategy.

This allows performance evaluation against a passive benchmark.

---

## How to Run

1. Clone this repository
2. Install dependencies (see requirements.txt)
3. Run the script (main.py)

---

## Example Output

<img width="1106" height="544" alt="Screenshot 2025-07-15 at 7 45 09 pm" src="https://github.com/user-attachments/assets/c71e58b7-9acc-4500-8d3e-d37e5fd67bcf" />

---

## Possible Extensions (possibly implemented by me in future projects)

- Better optimise ARIMA parameters for a higher Sharpe ratio.
- Allow the user to **select an end date** rather than automatically end at the current date
- Calculate **annualised returns, volatility, and Sharpe ratio**.
- Include **transaction costs** to simulate real-world profitability.
- Apply the strategy to a portfolio of stocks with rebalancing.

---

## Purpose

This project was created in the early steps of my journey into Financial Analysis with Python. It demonstrates:

- ARIMA model implementation
- Predictive analysis strategy implementation
- Financial data analysis with Python
- Backtesting and performance evaluation

---

## Author

**Muhammad Muntasir Shahzad**  
Student at King's College London, University of London. Studying Mathematics with Management and Finance   
Graduating: Summer 2026  
[LinkedIn Profile](www.linkedin.com/in/muntasir-shahzad) | [Email](muntasir.s.2004@gmail.com)

Please don't hesitate to contact me if you have any questions, suggestions, or otherwise.

---

## Disclaimer

This code is for educational purposes only and does not constitute financial advice or an investment recommendation.
