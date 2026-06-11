import pandas as pd
students = {"name": ["Ali","Sara","Reza","Mina"],"score": [18,19,17,20]}
#------------------------------
df = pd.DataFrame(students)
print ("Dataset:\n", df)
print ("------------------------------")
print("Average Score:", df["score"].mean())
print ("------------------------------")
print("Top Student:\n", df[df["score"] == df["score"].max()])
print ("------------------------------")
print ("Top Scores:\n", df[df["score"] > 18])
