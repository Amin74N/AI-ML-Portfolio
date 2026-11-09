import pandas as pd
df = pd.read_csv("binary_classification_sample.csv")
from sklearn.model_selection import train_test_split
x = df.drop(columns = ['Purchased'])
y = df['Purchased']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify= y, random_state = 42)
print(df['Purchased'].value_counts())
