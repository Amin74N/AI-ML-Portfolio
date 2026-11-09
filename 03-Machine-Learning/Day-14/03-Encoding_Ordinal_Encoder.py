import pandas as pd
df = pd.read_csv("binary_classification_sample.csv")
pd.set_option('display.max_columns', None)
print(df)
print("---------------------------------------------------------------------")
from sklearn.preprocessing import OrdinalEncoder
df.dropna(inplace = True)
ord_encoder = OrdinalEncoder()
df['Education'] = ord_encoder.fit_transform(df[['Education']]).astype(int)
df['Department'] = ord_encoder.fit_transform(df[['Department']]).astype(int)
print(df)
