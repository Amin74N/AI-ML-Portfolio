# Importing data:
import pandas as pd
df = pd.read_csv('emails.csv')
#---------------------------------------------------------------------
# Making new feature:
df['word_count'] = df['text'].fillna('').apply(lambda x: len(x.split()))
#---------------------------------------------------------------------
# Showing raw data:
print("Dataset:\n", df)
#---------------------------------------------------------------------
# Target/Feature split:
X = df[['word_count']]
y = df.label
#---------------------------------------------------------------------
# Train/Test split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
#---------------------------------------------------------------------
# Model building:
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
#---------------------------------------------------------------------
# Model training:
model.fit(X_train, y_train)
#---------------------------------------------------------------------
# Model prediction + probability:
probpredictions = model.predict_proba(pd.DataFrame([[len("free money prize".split())]], columns=['word_count']))
print("Probability of spam email is:")
print(probpredictions, "[[NotSpam Spam]]")
#---------------------------------------------------------------------
# Accuracy measurement:
predictions = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy Score:", accuracy)
#---------------------------------------------------------------------
# ConfusionMatrix:
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)
