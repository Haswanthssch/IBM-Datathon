import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml

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

            # Fit the Random Forest model using a PMML Pipeline
            pipeline = PMMLPipeline([("regressor", RandomForestRegressor(n_estimators=100, random_state=42))])
            pipeline.fit(X_train, y_train)

            # Save the model as PMML
            pmml_filename = f"{city}_food_wastage_model.pmml"
            sklearn2pmml(pipeline, pmml_filename, with_repr=True)
            print(f"Model saved as {pmml_filename} for city {city}")

        else:
            print(f"Not enough data for city {city} to train a model.")

    except Exception as e:
        print(f"Error processing city {city}: {e}")
