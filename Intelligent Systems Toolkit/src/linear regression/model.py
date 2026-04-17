import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn,linear_model import LinearRegression
#from sklearn.metrics import r2_score

plt.style.use('ggplot')
sns.set_style('whitegrid')

class LinearRegression:
    def __init__(self):
        self.slope = None # 'b' from y = a + bx
        self.intercept = None # 'a' from y = a + bx
        self.r_squared = None # R^2 score

    def fit(self, X, y):
        """
        b = Sxy / Sxx
        a = ȳ - b·x̄
        """
        # means
        x_mean = np.mean(X)
        y_mean = np.mean(y)

        # Sxx and Sxy
        Sxx = np.sum((X - x_mean) ** 2)
        Sxy = np.sum((X - x_mean) * (y - y_mean))

        # slope and intercept
        self.slope = Sxy / Sxx
        self.intercept = y_mean - self.slope * x_mean

        # R^2 score
        y_pred = self.predict(X)
        ss_total = np.sum((y - y_mean) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        self.r_squared = 1 - (ss_residual / ss_total)

        return self
    
    def predict(self, X):
        """y = a + bx
        """
        return self.intercept + self.slope * X

# Testing

attendance = np.array([10, 20, 30, 40, 50])
marks = np.array([50, 60, 70, 80, 90])

model = LinearRegression()
model.fit(attendance, marks)

print(f"Slope: {model.slope}")
print(f"Intercept: {model.intercept}")
print(f"R^2 Score: {model.r_squared}")
print(f"Formula: Marks = {model.intercept:.2f} + {model.slope:.2f} * Attendance")

# Predict 75% attendance
pred_75 = model.predict(np.array([75]))
print(f"Predicted Marks for 75% Attendance: {pred_75[0]:.2f}")