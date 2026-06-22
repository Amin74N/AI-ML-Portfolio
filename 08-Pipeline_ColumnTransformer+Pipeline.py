import pandas as pd
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv("covid_toy.csv")
pd.set_option('display.max_columns', None)
print(df)
print("---------------------------------------------------------------------")
df['has_covid'] = df['has_covid'].map({'Yes':1, "No":0})
print(df)
print("---------------------------------------------------------------------")
x = df.drop('has_covid', axis = 1)
y = df['has_covid']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
print("Train/Test Data Split")
print("---------------------------------------------------------------------")
numerical_features = ['age','fever']
categorical_features = ['gender', 'cough', 'city']
preprocessor = ColumnTransformer(transformers =
                                       [('num', StandardScaler(), numerical_features),
                                        ('cat', OneHotEncoder(handle_unknown= 'ignore'), categorical_features),])
pipeline = pipeline.Pipeline(steps = [('preprocessing',preprocessor),
                                      ('classifier', RandomForestClassifier(random_state=42))])
print("Pipeline Built")
print("---------------------------------------------------------------------")
pipeline.fit(x_train, y_train)
print("Model Trained")
print("---------------------------------------------------------------------")
y_pred = pipeline.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("---------------------------------------------------------------------")
def predict_covid(pipeline, new_data):
    input_df = pd.DataFrame([new_data])
    prediction = pipeline.predict(input_df)
    return "Has COVID" if prediction == 1 else "No COVID"
new_input = {"age": 40, "gender": "Female", "fever": 102.0, "cough": "Mild", "city": "Mumbai"}
print(predict_covid(pipeline, new_input))
