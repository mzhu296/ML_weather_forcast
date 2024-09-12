```markdown
# Weather Forecast Model

Welcome to my weather forecast project! 🎉 This repo has a machine learning model that predicts the maximum temperature for the next ten days using historical weather data. (low accuracy tho)

## What's in the Repo

- **weather_predict.py**: The script that trains the model and makes predictions.
- **weather_data.csv**: Sample historical weather data used for training the model.
- **weather_forecast_model.pkl**: The trained model saved using joblib.

## How It Works

1. **Data**: The model uses weather data with columns like longitude, latitude, year, month, day, and temperature readings.
2. **Model**: I’m using a RandomForestRegressor to predict the maximum temperature. It’s trained on historical data and then used to forecast future temperatures.
3. **Prediction**: The model predicts temperatures for the next ten days based on historical trends.

## Running the Script

1. **Clone the Repo**:
   ```sh
   git clone https://github.com/mzhu296/ML_weather_forecast.git
   ```

2. **Go to the Project Directory**:
   ```sh
   cd ML_weather_forecast
   ```

3. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Script**:
   ```sh
   python weather_predict.py
   ```

5. **Check the Results**: The script will print the predictions and show a plot of historical vs. predicted temperatures.

## Dependencies

You’ll need these Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

## License

This project is under the MIT License. Feel free to use and modify it as you like!

```

Let me know if there’s anything specific you’d like to add or tweak!
