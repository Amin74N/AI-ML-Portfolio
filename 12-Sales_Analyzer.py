import pandas as pd
df = pd.read_csv('sales.csv')
#------------------------------
print ("Dataset:\n", df)
print ("------------------------------")
print("Total Sales:", df["sales"].sum())
print ("------------------------------")
print("Average Sale:", df["sales"].mean())
print ("------------------------------")
print("Best Product:\n", df[df["sales"] == df["sales"].max()])
print ("------------------------------")
print ("Sorted Products By Sales:\n", df.sort_values("sales", ascending=False))
