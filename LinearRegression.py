import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "dhrubangtalukdar/fortune-500-companies-stock-data")

print("Path to dataset files:", path)
