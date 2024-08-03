import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

# Load dataset
filepath = 'D://OpenAQtask/pm25data.csv'
data = pd.read_csv(filepath)

print(data[:3])
# Preprocess data
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Store original data for plotting
original_data = data.copy()

data.drop(columns=['location_id','sensors_id','location','lat','lon','units'], inplace=True)

# Interpolate missing values
data['value'] = data['value'].interpolate(method='nearest')

# Handle outliers
mean_value = data['value'].mean()
std_value = data['value'].std()
median_value = data['value'].median()
data['value'] = np.where(abs(data['value'] - mean_value) > 3 * std_value, median_value, data['value'])


# Calculate split points for 80-10-10 ratio
total_rows = len(data)
train_end = int(total_rows * 0.8)
val_end = int(total_rows * 0.9)


# Split data
train_data = data.iloc[:train_end]
val_data = data.iloc[train_end:val_end]
test_data = data.iloc[val_end:]

print(train_data[:3])

# 1. Use auto_arima to find the best p, d, q values on the training data
model = auto_arima(train_data['value'], seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
model_fit = model.fit(train_data['value'])

# 2. Validate on the validation set
val_forecast = model_fit.predict(n_periods=len(val_data))
val_index = val_data.index
val_mse = mean_squared_error(val_data['value'], val_forecast)
print(f'Validation Mean Squared Error: {val_mse}')

# 3. Forecasting on the test set
forecast = model_fit.predict(n_periods=len(test_data))
forecast_index = test_data.index
test_mse = mean_squared_error(test_data['value'], forecast)
print(f'Test Mean Squared Error: {test_mse}')

# 4. Plot Results
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['value'], label='Training Data', color='blue')
plt.plot(val_data.index, val_data['value'], label='Validation Data', color='orange')
plt.plot(test_data.index, test_data['value'], label='Test Data', color='green')
plt.plot(forecast_index, forecast, label='Forecast', color='red')
plt.xlabel('Date Time')
plt.ylabel('PM2.5 (µg/m³)')
plt.title('Training, Validation, Test Data and Forecast')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

future_start_date = pd.to_datetime('2021-01-01')
future_end_date = pd.to_datetime('2022-01-01')
number_of_future_steps = int((future_end_date - future_start_date).total_seconds() / 3600)  # For hourly data

future_forecast = model_fit.predict(n_periods=number_of_future_steps)
future_forecast_index = pd.date_range(start=future_start_date, periods=number_of_future_steps, freq='h')

# Plot future forecast
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['value'], label='Original Data', color='blue')
plt.plot(future_forecast_index, future_forecast, label='Future Forecast', color='red')
plt.xlabel('Date Time')
plt.ylabel('PM2.5 (µg/m³)')
plt.title('Original Data and Future Forecast')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()