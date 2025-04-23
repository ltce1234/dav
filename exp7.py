import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv('Electric_Production.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

ts = df['IPG2211A2N']

model = ARIMA(ts, order=(1, 1, 1))
model_fit = model.fit()

print(model_fit.summary())

forecast_steps = 12
forecast = model_fit.forecast(steps=forecast_steps)

future_dates = pd.date_range(start=ts.index[-1], periods=forecast_steps + 1, freq='MS')[1:]

plt.figure(figsize=(12, 5))
plt.plot(ts, label='Observed')
plt.plot(future_dates, forecast, label='Forecast', color='red', linestyle='--')
plt.title('Electricity Production Forecast (ARIMA)')
plt.xlabel('Year')
plt.ylabel('Production Value')
plt.legend()
plt.grid(True)
plt.show()
