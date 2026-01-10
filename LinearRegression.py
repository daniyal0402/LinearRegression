import kagglehub
import os
import pandas as pd

# Download latest version
path = kagglehub.dataset_download(
    "dhrubangtalukdar/fortune-500-companies-stock-data")

file_path = os.path.join(path, "fortune_500_stock_data.csv")
df = pd.read_csv(file_path)


print(df.head())
print(df.info())
# print("Path to dataset files:", path)
# print(os.listdir(path))

# inner_dir = os.path.join(path, "fortune_500_stock_data")
# print("Inner files:", os.listdir(inner_dir))
