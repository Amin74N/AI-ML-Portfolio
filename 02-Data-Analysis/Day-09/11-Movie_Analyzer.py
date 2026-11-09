import pandas as pd
df = pd.read_csv('movies.csv')
#------------------------------
print ("Dataset:\n", df)
print ("------------------------------")
print("Average Rate:", df["rating"].mean())
print ("------------------------------")
print("Best Movie:\n", df[df["rating"] == df["rating"].max()])
print ("------------------------------")
print ("Top Rated Movies:\n", df[df["rating"] > 8])
