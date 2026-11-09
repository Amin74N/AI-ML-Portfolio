import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("movies.csv")
sns.barplot(x = 'movie', y= 'rating', data= df)
plt.show()
sns.histplot(x= 'rating', data = df, kde = True)
plt.show()
sns.boxplot(x= 'rating', data = df)
plt.show()
sns.heatmap(df.corr(numeric_only= True), annot=True)
plt.show()
