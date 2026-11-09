# Data Loading:
import pandas as pd
df = pd.read_csv("customer_clean.csv")
# ----------------------------------------------------------------------------------------------------------------------
# Exploratory Data Analysis (EDA):
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('dark')
sns.set_context('paper', font_scale= 1.3)
plt.figure(figsize = (8,5))
sns.histplot(df["income"], color= 'blue')
plt.show()
sns.countplot(data= df, x= "city", hue= "purchased", palette= 'afmhot')
plt.show()
sns.heatmap(df.corr(numeric_only= True),annot= True, cmap= 'Greens', linecolor= 'white', linewidths= 1)
plt.show()
