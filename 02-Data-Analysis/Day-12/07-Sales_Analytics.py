import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("sales.csv")
sns.barplot(x = 'month', y= 'sales', data= df)
plt.show()
sns.histplot(x= 'sales', data = df, kde = True)
plt.show()
sns.boxplot(x= 'sales', data = df)
plt.show()
sns.heatmap(df.corr(numeric_only= True), annot=True)
plt.show()
