# Data Loading:
import pandas as pd
df = pd.read_csv("customer_raw.csv")
print("Raw Dataset:")
print(df, "\n---------------------------------------------------------------------")
print("Raw Dataset Information:")
print(df.info(), "\n---------------------------------------------------------------------")
print("Raw Dataset Description:")
print(df.describe(), "\n---------------------------------------------------------------------")
# ----------------------------------------------------------------------------------------------------------------------
# Data Cleaning:
print("Raw Dataset Missing Values:")
print(df.isnull().sum(), "\n---------------------------------------------------------------------")
df.fillna(df.mean(numeric_only=True), inplace=True)
df.dropna(subset=["city"], inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(inplace=True)
print("Clean Dataset:")
print(df, "\n---------------------------------------------------------------------")
print("Clean Dataset Information:")
print(df.info(), "\n---------------------------------------------------------------------")
print("Clean Dataset Description:")
print(df.describe())
df.to_csv("customer_clean.csv",index=False)
