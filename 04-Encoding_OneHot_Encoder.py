import pandas as pd
df = pd.read_csv("binary_classification_sample.csv")
pd.set_option('display.max_columns', None)
print(df)
print("---------------------------------------------------------------------")
from sklearn.preprocessing import OneHotEncoder
df.dropna(inplace= True)
encoder = OneHotEncoder(drop = 'first', sparse_output= False)
col = ['Department']
encoded_col = encoder.fit_transform(df[col])
encoded_df = pd.DataFrame(encoded_col, columns = encoder.get_feature_names_out(col)).astype(int)
encoded_df.index = df.index
final_df = pd.concat([df, encoded_df], axis = 1)
final_df.drop(columns= 'Department', inplace = True)
print(final_df)
