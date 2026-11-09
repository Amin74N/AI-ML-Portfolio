# Importing data:
import pandas as pd
music_data = pd.read_csv('music.csv')
#---------------------------------------------------------------------
# Showing raw data:
print(music_data)
#---------------------------------------------------------------------
# Target/Feature split:
X = music_data.drop(columns= ['genre'])
y = music_data['genre']
#---------------------------------------------------------------------
# Train/Test split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#---------------------------------------------------------------------
# Model building:
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
#---------------------------------------------------------------------
# Model training:
model.fit(X_train, y_train)
#---------------------------------------------------------------------
# Model prediction:
predictions = model.predict(X_test)
#---------------------------------------------------------------------
# Accuracy measurement:
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, predictions)
#---------------------------------------------------------------------
# Showing the Result:
print(score)
