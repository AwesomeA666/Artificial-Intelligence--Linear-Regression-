# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Manual data: Years and corresponding prices of Mahindra Thar
years = [2020, 2021, 2022]  # Years with known prices
prices = [980000, 1125000, 1359000]  # Corresponding prices of Mahindra Thar

# Convert data into pandas DataFrame
df = pd.DataFrame({'Year': years, 'Price': prices})

# Preprocess the data
X = df['Year'].values.reshape(-1, 1)  # Feature: Year
y = df['Price'].values  # Target: Price

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict prices for the next 3 years
future_years = [2023, 2024, 2025]
future_X = np.array(future_years).reshape(-1, 1)
predicted_prices = model.predict(future_X)

# Print predicted prices for the next 3 years
print("Predicted prices for the next 3 years:")
for year, price in zip(future_years, predicted_prices):
    print(f"Year {year}: Rs. {price:.2f}")

# Plot the results
plt.scatter(X, y, color='blue', label='Actual Prices')
plt.plot(np.concatenate((X, future_X)), np.concatenate((model.predict(X), predicted_prices)), color='red', linestyle='--', label='Predicted Prices')
plt.title('Mahindra Thar Price Prediction')
plt.xlabel('Year')
plt.ylabel('Price (Rs.)')
plt.xticks(np.arange(min(years)-1, max(future_years)+2, 1))
plt.legend()
plt.grid(True)
plt.show()
