import kagglehub
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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


df['Tomorrow_Open'] = df['Open'].shift(-1)
df['Close_Equals_Tomorrow_Open'] = df['Close'] == df['Tomorrow_Open']
df['Close_Equals_Tomorrow_Open'].value_counts()
df['Close_Open_Diff'] = df['Tomorrow_Open'] - df['Close']
df['Close_Open_Diff'].describe()

df['Close_Open_Almost_Equal'] = np.isclose(
    df['Close'],
    df['Tomorrow_Open'],
    atol=0.01
)

df['Close_Open_Almost_Equal'].value_counts()

df['Close_Open_Diff'].hist(bins=50)
plt.title('Tomorrow Open − Today Close')
plt.xlabel('Price Difference')
plt.ylabel('Frequency')
plt.show()
