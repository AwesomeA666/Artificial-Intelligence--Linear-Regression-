# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data: Runs scored by Virat Kohli in previous T20 World Cup editions
editions = ['2012', '2014', '2016', '2021', '2022']
runs = [185, 319, 273, 68, 296]  # Replace with actual runs data

# Convert data into pandas DataFrame
df = pd.DataFrame({'Edition': editions, 'Runs': runs})

# Preprocess the data
X = np.arange(len(editions)).reshape(-1, 1)  # Feature: Edition index
y = df['Runs'].values  # Target: Runs scored

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict runs for upcoming T20 World Cup editions (2024, 2026, 2028)
upcoming_editions = ['2024', '2026', '2028']
upcoming_X = np.arange(len(editions), len(editions) + len(upcoming_editions)).reshape(-1, 1)
predicted_runs = model.predict(upcoming_X)

# Print predicted runs for upcoming editions
print("Predicted runs for upcoming T20 World Cup editions:")
for i, runs in enumerate(predicted_runs, start=1):
    print(f"T20 World Cup {upcoming_editions[i-1]}: {int(runs)} runs")

# Plot the results
plt.scatter(X, y, color='blue', label='Actual Runs')
plt.plot(np.concatenate((X, upcoming_X)), np.concatenate((model.predict(X), predicted_runs)), color='red', linestyle='--', label='Predicted Runs')
plt.title('Virat Kohli T20 World Cup Runs Prediction')
plt.xlabel('Edition')
plt.ylabel('Runs')
plt.xticks(np.arange(len(editions) + len(upcoming_editions)), [f"T20 WC {year}" for year in editions + upcoming_editions])
plt.legend()
plt.grid(True)
plt.show()
