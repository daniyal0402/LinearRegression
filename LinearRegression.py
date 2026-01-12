import kagglehub
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

# Using Linear Regression
features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})

print(coefficients)

# print(df.head())
# print(df.info())
# print("Path to dataset files:", path)
# print(os.listdir(path))

# inner_dir = os.path.join(path, "fortune_500_stock_data")
# print("Inner files:", os.listdir(inner_dir))
