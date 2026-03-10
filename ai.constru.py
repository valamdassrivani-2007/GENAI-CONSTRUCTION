# AI BuildWise Construction Planning
# Predict construction time based on project data

import numpy as np
from sklearn . linear_model import LinearRegression
# Sample training data
# Columns: [Project Size (sq ft), Workers, Budget (lakhs)]
X = np.array([
    [1000, 10, 20],
    [1500, 12, 25],
    [2000, 15, 30],
    [2500, 18, 40],
    [3000, 20, 50]
])

# Output: Construction time in months
y = np.array([6, 7, 8, 10, 12])

# Create AI model
model = LinearRegression()

# Train model
model.fit(X, y)

# Input new project details
size = float(input("Enter project size (sq ft): "))
workers = int(input("Enter number of workers: "))
budget = float(input("Enter budget (lakhs): "))

# Prediction
prediction = model.predict([[size, workers, budget]])

print("Estimated Construction Time:", prediction[0], "months")