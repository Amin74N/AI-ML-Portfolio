import pandas as pd
df = pd.read_csv("binary_classification_sample.csv")
pd.set_option('display.max_columns', None)
print(df)
print("---------------------------------------------------------------------")
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])
df['Department'] = encoder.fit_transform(df['Department'])
df['Education'] = encoder.fit_transform(df['Education'])
print(df)
