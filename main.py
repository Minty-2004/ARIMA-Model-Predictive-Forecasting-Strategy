import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import time
import sys
import datetime
from datetime import date, timedelta
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import mean_squared_error
import itertools
import warnings
warnings.filterwarnings('ignore')


#Defining the SMAPE error function (Symmetric Mean Absolute Percentage Error)
def smape_kun(y_true, y_pred, epsilon=1e-8):
  #Calculate the denominator to avoid division by zero
  denominator = ((np.abs(y_pred) + np.abs(y_true))/2) + epsilon

  #Calculate the absolute percentage error with symetric scaling
  absolute_percentage_error = np.abs(y_pred - y_true) * 100 / denominator

  #Calculate the mean of the symmetric absolute percentage errors
  mean_smape = np.mean(absolute_percentage_error)

  return mean_smape


#User inputs stock data and timeframes
ticker = str(input("Please enter the ticker symbol you would like to analyse (for example, the ticker for Apple would be 'AAPL'): ")) #Choose any ticker
test_date = str(input("What date would you like to start the analysis from? Only the tail 20% will be used to test the model. Please use the format YYYY-MM-DD: ")) #Creating time frame for analysis
data = yf.download(ticker, start=test_date, end=datetime.datetime.today().strftime('%Y-%m-%d')) #Downloading stock data from yfinance
if data.empty:
    sys.exit("Could not download data for the specified ticker and date range. Please check the ticker symbol and start date.") #Contingency message if ticker/date entered is invalid
df = data.iloc[(int(0.8*len(data))-1):,:] #Copy of test data (with one extra entry) for comparative analysis later
print()
# Calculate and import train data, plot
train_data = data.iloc[0:int(0.8*len(data)),:]
test_data = data.iloc[int(0.8*len(data)):,:]

train_series = train_data['Close']
test_series = test_data['Close']

plt.figure(figsize=(12,6))
plt.plot(train_data["Close"], label="Training Data", color="blue")
plt.plot(test_data["Close"], label="Testing Data", color="green")
plt.title(f"{ticker} Close Prices, Training and Testing Data")
plt.xlabel("Date")
plt.ylabel("Close Prices")
plt.legend()
plt.show()


#Apply (possibly repeated) DF Test to find d
d = 0
train_diff = train_series #Introducing new train_diff variable to difference on, while maintaining original series for the ARIMA Modelling

dftest = adfuller(train_diff)
while dftest[1] > 0.05: #Continually differences until the data is deemed stationary at a 5% significance level
    d += 1
    train_diff = train_diff.diff(periods=1).dropna()
    dftest = adfuller(train_diff, autolag="AIC")


#Calculating AIC values to find optimal p and q parameters
p_range = range(0, 3)  # Test p values from 0 to 2
q_range = range(0, 3)  # Test q values from 0 to 2
best_aic = np.inf  # Initialize with worst possible AIC
best_order = (0, d, 0)  # Default order

# Grid search over p and q combinations
for p, q in itertools.product(p_range, q_range):
  try:
    # Initialize ARIMA model
    model = ARIMA(train_series, order=(p, d, q))
    # Fit model to training data
    results = model.fit()
    # Check if current model is better
    if results.aic < best_aic:
      # Update best AIC
      best_aic = results.aic
      # Update best order
      best_order = (p, d, q)
  except:
    # Skip invalid parameter combinations
    continue
p = best_order[0]
q = best_order[2]


#Carrying out the ARIMA Model
print()
print(f"Optimal parameters for the ARIMA Model are estimated to be p={p}, d={d}. q={q}")
if p != 0 and q != 0:
  print("Running model now. Please allow up to 10 minutes...")
else:
  print("Runing model now. Please allow some time...")

history = [x[0] for x in train_series.values] #Initialise the history with the training data
predictions = list() #Empty list, will append predictions here as the model runs

for t in range(len(test_series)): # Iterate through the test data points
  model = ARIMA(history, order=(p,d,q))
  model_fit = model.fit() #Fitting the model onto the data

  output = model_fit.forecast()
  yhat = output[0] #Extracts the prediction for the next day
  predictions.append(yhat) #Adds next day prediction to the predictions list

  obs = test_series.values[t][0] #After prediction made, extract the actual expected value
  history.append(obs) #Append the actual value into the history (as the prediction has been made for the day) so that the model can learn from any mistakes and adjust future forecasts)

# Convert test_series and predictions to list formats before calculating SMAPE
test_series_np = test_series.values
predictions_np = np.array(predictions)

# Calculate error margins to determine how well the model performed
mse = mean_squared_error(test_series_np, predictions_np)
error_smape = smape_kun(test_series_np, predictions_np)
print()
print("The predictive analysis has been completed.")
print("Mean Squared Error: %.3f" % mse)
print("SMAPE Error: %.3f" % error_smape)
print()
print("Creating graphs for visual results...")
print()
time.sleep(3) #Wait 3 second before executing remaining lines of code, allow user to read the information provided


#Plotting with full history, including the training data
print("The following graph depicts the predictions, in red, plotted against the training and testing data.")
time.sleep(2)
predictions = pd.Series(predictions, index=test_series.index)
history = pd.Series(history, index=train_series.index.union(test_series.index))
plt.figure(figsize=(12,7))
plt.title(f"{ticker} Close Prices")
plt.xlabel("Date")
plt.ylabel("Close Prices")
plt.plot(train_series, color="blue")
plt.plot(test_series, color="green", marker=".", label="Actual Closing Prices")
plt.plot(predictions, color="red", linestyle="--", label="Forcast Closing Prices")
plt.legend()
plt.show()
print()
time.sleep(3)

#Plotting only test data
print("The following graph provides a zoomed snapshot of the predictions, in red, plotted against the testing data alone.")
time.sleep(2)
plt.figure(figsize=(12,7))
plt.title(f"{ticker} Close Prices")
plt.xlabel("Date")
plt.ylabel("Close Prices")
plt.plot(test_series, color="green", marker=".", label="Actual Closing Prices")
plt.plot(predictions, color="red", linestyle="--", label="Forcast Closing Prices")
plt.legend()
plt.show()
time.sleep(3)

#Calculating Cumulative Profits using ARIMA signals
print()
print()
print("Now comparing the ARIMA signals model against the benchmark Buy and Hold Strategy")
time.sleep(2)
df["Predictions"] = predictions
df["PredictionsDiff"] = df["Predictions"].diff()
df["Signal"] = (df["PredictionsDiff"] > 0).astype(int)
df["CloseDiff"] = df["Close"].diff()
df["Profits"] = df["Signal"] * df["CloseDiff"]
df["StratWealth"] = df["Profits"].cumsum()
df["BHWealth"] = df["CloseDiff"].cumsum()
df = df.dropna()
df['StratWealth'].plot(label='ARIMA Model Strategy')
df['BHWealth'].plot(label='Buy & Hold')
plt.legend()
plt.title(f"Total returns of {ticker} with the Positive 50-day Momentum Strategy are {round(df['StratWealth'].iloc[-1],2)}, compared to Buy & Hold returns of {round(df['BHWealth'].iloc[-1],2)}")
plt.show()