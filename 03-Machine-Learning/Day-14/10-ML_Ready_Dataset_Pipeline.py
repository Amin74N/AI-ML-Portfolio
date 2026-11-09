import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
#---------------------------------------------------------------------
# Importing data:
df = pd.read_csv("customer_dataset.csv")
print("Raw Data:\n", df)
#---------------------------------------------------------------------
# Target / Feature split:
X = df.drop(columns= ['purchased'])
y = df['purchased'].map({'Yes': 1, 'No': 0})
#---------------------------------------------------------------------
# Columns:
num_cols = ['age', 'income']
cat_cols = ['name', 'gender']
#---------------------------------------------------------------------
# Numerical pipeline:
num_pipeline = Pipeline([('scaler', StandardScaler())])
#---------------------------------------------------------------------
# Categorical pipeline:
cat_pipeline = Pipeline([('encoder', OneHotEncoder(handle_unknown= 'ignore'))])
#---------------------------------------------------------------------
# Combine pipelines:
preprocessor = ColumnTransformer([('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])
#---------------------------------------------------------------------
# Final pipeline:
pipeline = Pipeline([('preprocessing', preprocessor)])
#---------------------------------------------------------------------
# Apply pipeline:
X_processed = pipeline.fit_transform(X)
#---------------------------------------------------------------------
# Showing the result:
print("\nProcessed Data:\n", X_processed)
