import pandas as pd
df = pd.read_csv('employees.csv')
#------------------------------
print ("Dataset:\n", df)
print ("------------------------------")
print("Average Salary:", df["salary"].mean())
print ("------------------------------")
print("Highest Salary:\n", df[df["salary"] == df["salary"].max()])
print ("------------------------------")
print ("Top Salaries:\n", df[df["salary"] > 3500])
print ("------------------------------")
print ("Sorted Salaries:\n", df.sort_values("salary", ascending=False))
