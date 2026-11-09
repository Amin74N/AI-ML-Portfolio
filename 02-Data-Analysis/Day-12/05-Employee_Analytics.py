import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("employees.csv")
sns.barplot(x = 'name', y= 'salary', data= df)
plt.show()
sns.histplot(x= 'salary', data = df, kde = True)
plt.show()
sns.boxplot(x= 'salary', data = df)
plt.show()
sns.heatmap(df.corr(numeric_only= True), annot=True)
plt.show()
