import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#-----------------------------------------------------------------------------
# Datasets:
crash_df = sns.load_dataset('car_crashes')
tips_df = sns.load_dataset('tips')
#-----------------------------------------------------------------------------
# Distribution Plot:
sns.displot(crash_df['not_distracted'], kde=True, bins=25)
plt.show()
#-----------------------------------------------------------------------------
# Joint Plot:
sns.jointplot(x= 'speeding', y= 'alcohol', data= crash_df, kind= 'hex')
plt.show()
#-----------------------------------------------------------------------------
# KDE Plot:
sns.kdeplot(crash_df['alcohol'])
plt.show()
#-----------------------------------------------------------------------------
# Pair Plot:
sns.pairplot(tips_df, hue= 'sex', palette= 'Greens')
plt.show()
#-----------------------------------------------------------------------------
# Rug Plot:
sns.rugplot(tips_df['tip'])
plt.show()
#-----------------------------------------------------------------------------
# Styling:
sns.set_style('ticks')
sns.set_context('paper', font_scale= 1.5)
fig = sns.jointplot(x= 'speeding', y= 'alcohol', data= crash_df, kind= 'reg')
fig.figure.set_size_inches(8,4.5)
sns.despine(left=True, bottom=False, right=True, top=True)
plt.show()
