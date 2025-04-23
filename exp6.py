import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. Load the dataset
df = pd.read_csv('Electric_Production.csv')

# 2. Convert 'DATE' to datetime and set as index
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

# 3. Extract the target time series
ts = df['IPG2211A2N']

# 4. Plot the original time series
plt.figure(figsize=(12, 5))
plt.plot(ts, label='Electricity Production')
plt.xlabel('Year')
plt.ylabel('Production Value')
plt.title('Electricity Production Over Time')
plt.legend()
plt.grid(True)
plt.show()

# 5. Check for stationarity using Augmented Dickey-Fuller Test
adf_result = adfuller(ts)
print("Augmented Dickey-Fuller Test:")
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
for key, value in adf_result[4].items():
    print(f"Critical Value ({key}): {value}")

if adf_result[1] > 0.05:
    print("\nResult: The series is likely **not stationary** (p > 0.05).")
else:
    print("\nResult: The series is **stationary** (p <= 0.05).")

# 6. Difference the series (to make it stationary)
ts_diff = ts.diff().dropna()

# 7. Plot the differenced series
plt.figure(figsize=(12, 5))
plt.plot(ts_diff, label='Differenced Series', color='orange')
plt.xlabel('Year')
plt.ylabel('Differenced Production Value')
plt.title('Differenced Electricity Production Time Series')
plt.legend()
plt.grid(True)
plt.show()

# 8. Plot ACF and PACF of the differenced series
plt.figure(figsize=(10, 4))
plot_acf(ts_diff, lags=40)
plt.title("ACF of Differenced Series")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plot_pacf(ts_diff, lags=40, method='ywm')
plt.title("PACF of Differenced Series")
plt.tight_layout()
plt.show()
