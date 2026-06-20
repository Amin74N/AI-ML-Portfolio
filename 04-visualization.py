import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#--------------------------------------------------------------------
df = pd.read_csv("clean_sales.csv")
#--------------------------------------------------------------------
sns.barplot(x= "product",y= "quantity",data= df)
plt.savefig('sales_by_product.png', dpi = 300)
plt.show()
#--------------------------------------------------------------------
sns.histplot(df["price"], kde= True)
plt.savefig('price_distribution.png', dpi = 300)
plt.show()
#--------------------------------------------------------------------
sns.boxplot(y= df["rating"])
plt.savefig('rating_boxplot.png', dpi = 300)
plt.show()
#--------------------------------------------------------------------
sns.heatmap(df.corr(numeric_only= True),annot=True)
plt.savefig('correlation_heatmap.png', dpi = 300)
plt.show()
