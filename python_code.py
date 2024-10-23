import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_excel('fs1.xlsx', engine='openpyxl')

# Create a new column for combined figures
data['combined figures (kg/capita/year)'] = data['Food service estimate (kg/capita/year)']

# Handle missing values
required_columns = ['combined figures (kg/capita/year)', 'Food service estimate (kg/capita/year)']
if 'Hotel Name' in data.columns:
    required_columns.append('Hotel Name')
data.dropna(subset=required_columns, inplace=True)

# Limit the number of rows to avoid overflow
limited_rows = min(len(data), 10000)
data = data.head(limited_rows)

# Create a synthetic 'Date' column assuming daily data
data['Date'] = pd.date_range(start='2023-01-01', periods=len(data), freq='D')

# Set the 'Date' column as the index
data.set_index('Date', inplace=True)

# Dictionary to store predictions for each city
city_predictions = {}

# Group the data by 'city'
grouped = data.groupby('city')

# Loop over each city, train a Random Forest model, and predict
for city, group in grouped:
    try:
        # Ensure proper data types for interpolation
        group = group.infer_objects()
        
        # Convert specific columns to numeric
        group['combined figures (kg/capita/year)'] = pd.to_numeric(group['combined figures (kg/capita/year)'], errors='coerce')
        group['Food service estimate (kg/capita/year)'] = pd.to_numeric(group['Food service estimate (kg/capita/year)'], errors='coerce')

        # Fill missing values using interpolation
        group.interpolate(method='linear', inplace=True)

        # Check if there is enough data for the city
        if len(group) > 1:
            # Add lagged features
            group['lag_1'] = group['combined figures (kg/capita/year)'].shift(1)
            group['lag_2'] = group['combined figures (kg/capita/year)'].shift(2)

            # Drop rows with NaN values due to lagging
            group.dropna(inplace=True)
            
            # Add a 'day' feature to act as the input for the model
            group['day'] = range(len(group))

            # Features
            X = group[['day', 'Food service estimate (kg/capita/year)', 'lag_1', 'lag_2']]

            # Target
            y = group['combined figures (kg/capita/year)']

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Fit the Random Forest model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Predict for the next ten days
            next_day = pd.DataFrame({
                'day': [len(group)],  # Predict for the next day
                'Food service estimate (kg/capita/year)': [group['Food service estimate (kg/capita/year)'].mean()],
                'lag_1': [y.iloc[-1]],  # Last known wastage as lag_1
                'lag_2': [y.iloc[-2]]  # Second last known wastage as lag_2
            })

            forecast = []
            if 'Hotel Name' in group.columns:
                hotel_names = group['Hotel Name'].unique()  # Get unique hotel names for the city
            else:
                hotel_names = [f"Hotel_{i+1}" for i in range(10)]  # Use generic hotel labels if 'Hotel Name' column is absent
            
            for i in range(10):  # Forecast for 10 days to get more variation
                prediction = model.predict(next_day)[0]
                
                # Add some variation to the prediction
                noise = np.random.uniform(-0.05, 0.05) * prediction  # Random noise within ±5% of prediction
                prediction_with_variation = prediction + noise
                hotel_name = np.random.choice(hotel_names)  # Randomly pick a hotel name
                forecast.append((hotel_name, prediction_with_variation))  # Store hotel name and prediction

                # Update lags for the next prediction
                next_day['lag_1'] = [prediction_with_variation]
                next_day['lag_2'] = [next_day['lag_1'].iloc[0]]
                next_day['day'] += 1

            # Sort the forecast by predicted availability in descending order and get the top 3
            top_3_predictions = sorted(forecast, key=lambda x: x[1], reverse=True)[:3]
            city_predictions[city] = top_3_predictions

        else:
            if 'Hotel Name' in group.columns:
                hotel_name = group['Hotel Name'].iloc[0]
            else:
                hotel_name = "Generic_Hotel"
            city_predictions[city] = [(hotel_name, group['combined figures (kg/capita/year)'].mean())] * 3

    except Exception as e:
        print(f"Error processing city {city}: {e}")

# Print all city predictions
print("\nPredictions for all cities:")
for city, forecasts in city_predictions.items():
    print(f"City: {city}")
    for hotel, availability in forecasts:
        print(f"Hotel: {hotel}, Predicted availability: {availability:.2f} kg")

# Get user input for the number of people and city information
num_people = int(input("\nEnter the number of people: "))
city_name = input("Enter the city name: ").strip()

# Check for the city's predictions
if city_name in city_predictions:
    required_food = num_people * 1  # Assume 1 kg per person
    available_food = city_predictions[city_name]

    # Compare availability and assign up to the top 3 hotels
    assigned_count = 0
    for hotel, available in available_food:  # Use the top 3 predictions
        if available >= required_food:
            print(f"Assigning to {hotel} in {city_name}. Available food: {available:.2f} kg.")
            assigned_count += 1
        else:
            print(f"Not enough food in {hotel}. Required: {required_food} kg, Available: {available:.2f} kg.")

        # Stop after assigning 3 hotels
        if assigned_count == 3:
            break

    if assigned_count == 0:
        print(f"Not enough food available in {city_name}.")

else:
    print(f"No data available for city: {city_name}.")
