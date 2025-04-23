import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

co2_ts = sm.datasets.co2.load_pandas().data['co2']
co2_ts = co2_ts.interpolate(method='linear')
co2_ts.name = 'CO2 Levels'

fig_acf, ax_acf = plt.subplots(figsize=(12, 5))
plot_acf(co2_ts, lags=48, ax=ax_acf)
ax_acf.set(title='Autocorrelation Function (ACF) - CO2 Levels', xlabel='Lag (Months)', ylabel='ACF')

fig_pacf, ax_pacf = plt.subplots(figsize=(12, 5))
plot_pacf(co2_ts, lags=48, method='ols', ax=ax_pacf)
ax_pacf.set(title='Partial Autocorrelation Function (PACF) - CO2 Levels', xlabel='Lag (Months)', ylabel='PACF')

plt.tight_layout()
plt.show()

co2_diff = co2_ts.diff().dropna()

model = ARIMA(co2_ts, order=(1, 1, 1))
fitted_model = model.fit()

print(fitted_model.summary())

forecast = fitted_model.forecast(steps=12)

plt.plot(co2_ts, label='Actual CO2 Levels', color='blue')
plt.plot(pd.date_range(co2_ts.index[-1], periods=13, freq='M')[1:], forecast, label='Forecasted CO2 Levels', color='red')
plt.title('ARIMA Forecast for CO2 Levels')
plt.xlabel('Date')
plt.ylabel('CO2 Levels')
plt.legend()
plt.show()