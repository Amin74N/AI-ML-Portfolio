import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#---------------------------------------------------------------------------------------------
# Datasets:
tips_df = sns.load_dataset('tips')
#---------------------------------------------------------------------------------------------
# Bar Plot:
sns.barplot(x = 'sex', y= 'total_bill', data= tips_df, estimator= np.cov)
plt.show()
#---------------------------------------------------------------------------------------------
# Count Plot:
sns.countplot(x = 'sex', data= tips_df)
plt.show()
#---------------------------------------------------------------------------------------------
# Box Plot:
sns.boxplot(x= 'day', y= 'total_bill', data= tips_df, hue= 'sex')
plt.show()
#---------------------------------------------------------------------------------------------
# Violin Plot:
sns.violinplot(x= 'day', y= 'total_bill', data= tips_df, hue= 'sex', split = True)
plt.show()
#---------------------------------------------------------------------------------------------
# Strip Plot:
plt.figure(figsize= (8,4))
sns.stripplot(x= 'day', y= 'total_bill', data= tips_df, hue= 'sex', jitter= True, dodge= True)
plt.show()
#---------------------------------------------------------------------------------------------
# Swarm Plot:
sns.swarmplot(x= 'day', y= 'total_bill', data= tips_df)
plt.show()
#---------------------------------------------------------------------------------------------
# Pallets:
plt.figure(figsize= (8,6.5))
sns.set_style('dark')
sns.set_context('talk')
sns.stripplot(x= 'day', y= 'total_bill', data= tips_df, hue= 'sex', palette= 'afmhot')
plt.legend(loc= 0) # Upper Right: 1 /Upper Left: 2 /Lower Left: 3 /Lower Right: 4
plt.show()
