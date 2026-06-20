import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Datasets:
tips_df = sns.load_dataset('tips')
#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Regression Plot-1:
sns.set_context("paper", font_scale=1.4)
fig = sns.lmplot(x= 'total_bill', y= 'tip', hue= 'sex', data = tips_df, markers= ['o','^'], scatter_kws= {'s': 100, 'linewidths': .5, 'edgecolor': 'w'})
fig.figure.set_size_inches(8,6)
plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Regression Plot-2:
sns.set_context("paper", font_scale=1.4)
sns.lmplot(x= 'total_bill', y= 'tip', col= 'sex', row= 'time', data = tips_df, height= 5, aspect= .7)
plt.show()
