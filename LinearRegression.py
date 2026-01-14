import kagglehub
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt

# Download latest version
path = kagglehub.dataset_download(
    "dhrubangtalukdar/fortune-500-companies-stock-data")

# Reading data
folder_path = os.path.join(path, "fortune_500_stock_data")
file_path = os.path.join(folder_path, "Apple.csv")
df = pd.read_csv(file_path)

# Cleaning Data
df = df.drop(index=0).reset_index(drop=True)
df['Date'] = pd.to_datetime(df['Date'])
numeric_cols = [
    'Open', 'High', 'Low',
    'Close', 'Adjusted_Close', 'Volume'
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.sort_values('Date').reset_index(drop=True)
df['Tomorrow_Close'] = df['Close'].shift(-1)
df = df.dropna().reset_index(drop=True)


# setting params
X = df[['Open', 'High', 'Low', 'Volume']]
y = df[['Tomorrow_Close']]

# print(df[features].corr())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)


# Using Linear Regression###############################
lin_model = Pipeline([
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])

lin_model.fit(X_train, y_train)

y_train_pred_lin = lin_model.predict(X_train)
y_test_pred_lin = lin_model.predict(X_test)

lin_train_mse = mean_squared_error(y_train, y_train_pred_lin)
lin_test_mse = mean_squared_error(y_test, y_test_pred_lin)

lin_train_r2 = r2_score(y_train, y_train_pred_lin)
lin_test_r2 = r2_score(y_test, y_test_pred_lin)

print("Linear Regression")
print("Test MSE:", lin_test_mse)
print("Test R²:", lin_test_r2)
# using ridge regression
ridge_model = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', RidgeCV(alphas=np.logspace(-4, 4, 50)))
])

ridge_model.fit(X_train, y_train)

y_train_pred_ridge = ridge_model.predict(X_train)
y_test_pred_ridge = ridge_model.predict(X_test)

ridge_train_mse = mean_squared_error(y_train, y_train_pred_ridge)
ridge_test_mse = mean_squared_error(y_test, y_test_pred_ridge)

ridge_train_r2 = r2_score(y_train, y_train_pred_ridge)
ridge_test_r2 = r2_score(y_test, y_test_pred_ridge)

print("\nRidge Regression")
print("Best alpha:", ridge_model.named_steps['ridge'].alpha_)
print("Test MSE:", ridge_test_mse)
print("Test R²:", ridge_test_r2)


# Plotting

plt.figure(figsize=(8, 5))
plt.plot(y_test.values[:100], label='Actual Tomorrow Close')
plt.plot(y_test_pred_lin[:100], label='Linear Prediction', linestyle='--')
plt.plot(y_test_pred_ridge[:100], label='Ridge Prediction', linestyle=':')

plt.plot(y_train.values[:100], label='Training Tomorrow Close')
plt.plot(y_train_pred_lin[:100], label='Linear Prediction', linestyle='--')
plt.plot(y_train_pred_ridge[:100], label='Ridge Prediction', linestyle=':')

plt.title('Tomorrow Close Prediction (First 100 Test Days)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# results = pd.DataFrame({
#     'Model': ['Linear Regression', 'Ridge Regression'],
#     'Train MSE': [lin_train_mse, ridge_train_mse],
#     'Test MSE': [lin_test_mse, ridge_test_mse],
#     'Train R²': [lin_train_r2, ridge_train_r2],
#     'Test R²': [lin_test_r2, ridge_test_r2]
# })

# print(results)

# lin_coef = lin_model.named_steps['linear'].coef_
# ridge_coef = ridge_model.named_steps['ridge'].coef_

# coef_df = pd.DataFrame({
#     'Feature': X.columns,
#     'Linear Coef': lin_coef,
#     'Ridge Coef': ridge_coef
# })

# print(coef_df)

# coef_df.set_index('Feature').plot(kind='bar', figsize=(8, 5))
# plt.title('Linear vs Ridge Coefficients')
# plt.ylabel('Coefficient Value (standardized)')
# plt.grid(True)
# plt.show()
