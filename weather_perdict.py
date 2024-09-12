import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
data = pd.read_csv('weather_data.csv')

# Inspect data
print(data.head())
print(data.info())
# Example: Fill missing values with the median of the column
data.fillna(data.median(), inplace=True)

# Convert date/time column to datetime
data['Date/Time'] = pd.to_datetime(data['Date/Time'])

# Extract date components
data['Year'] = data['Date/Time'].dt.year
data['Month'] = data['Date/Time'].dt.month
data['Day'] = data['Date/Time'].dt.day

# Adjust features for training
features = data[['Longitude (x)', 'Latitude (y)', 'Year', 'Month', 'Day']]
target = data['Max Temp (°C)']  # or another target variable

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
r2 = r2_score(y_test, y_pred)
print(f'R-squared Score: {r2}')


import joblib

# Save model
joblib.dump(model, 'weather_forecast_model.pkl')

# Load model
loaded_model = joblib.load('weather_forecast_model.pkl')
# Plot temperature trends
plt.figure(figsize=(12, 6))
plt.plot(data['Date/Time'], data['Max Temp (°C)'], label='Max Temperature')
plt.plot(data['Date/Time'], data['Min Temp (°C)'], label='Min Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Trends Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Distribution of maximum temperatures
plt.figure(figsize=(8, 6))
sns.histplot(data['Max Temp (°C)'], kde=True)
plt.xlabel('Max Temperature (°C)')
plt.title('Distribution of Max Temperatures')
plt.show()

# Generate future dates
future_dates = pd.date_range(start='2024-01-19', periods=10)

# Prepare future data
future_data = pd.DataFrame({
    'Date/Time': future_dates,
    'Year': future_dates.year,
    'Month': future_dates.month,
    'Day': future_dates.day,
    'Longitude (x)': -81.15,
    'Latitude (y)': 43.03
})

# Use the same feature columns as used in the model
future_features = future_data[['Longitude (x)', 'Latitude (y)', 'Year', 'Month', 'Day']]

# Predict weather
future_predictions = model.predict(future_features)
future_data['Predicted Max Temp (°C)'] = future_predictions

# Plot historical and predicted temperatures
plt.figure(figsize=(12, 6))
plt.plot(data['Date/Time'], data['Max Temp (°C)'], label='Historical Max Temperature')
plt.plot(future_data['Date/Time'], future_data['Predicted Max Temp (°C)'], label='Predicted Max Temperature', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Forecast for the Next Ten Days')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

