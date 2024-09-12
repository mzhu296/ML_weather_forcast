import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
data = pd.read_csv('weather_data.csv')

# Fill missing values with the median of the column
data.fillna(data.median(), inplace=True)

# Convert date/time column to datetime
data['Date/Time'] = pd.to_datetime(data['Date/Time'])

# Extract date components
data['Year'] = data['Date/Time'].dt.year
data['Month'] = data['Date/Time'].dt.month
data['Day'] = data['Date/Time'].dt.day

# Feature selection
features = data[['Longitude (x)', 'Latitude (y)', 'Year', 'Month', 'Day']]
target = data['Max Temp (°C)']  # or any other target variable you are interested in

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
r2 = r2_score(y_test, y_pred)
print(f'R-squared Score: {r2}')

# Save model
joblib.dump(model, 'weather_forecast_model.pkl')

# Load model
loaded_model = joblib.load('weather_forecast_model.pkl')

# Generate future dates
future_dates = pd.date_range(start='2024-09-11', periods=10)

# Create a DataFrame for future dates
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

# Predict weather for the next ten days
future_predictions = loaded_model.predict(future_features)

# Add predictions to the DataFrame
future_data['Predicted Max Temp (°C)'] = future_predictions

# Display the predicted data
print(future_data)

# Optional: Plot historical and predicted temperatures
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
