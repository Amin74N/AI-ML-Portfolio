import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#-------------------------------------------------------------------------------------------------------------------------------------------------
# Datasets:
crash_df = sns.load_dataset('car_crashes')
flights = sns.load_dataset('flights')
iris = sns.load_dataset('iris')
tips_df = sns.load_dataset('tips')
att_df = sns.load_dataset('attention')
#-------------------------------------------------------------------------------------------------------------------------------------------------
# Heatmaps-1:
plt.figure(figsize = (13,5))
sns.set_context('paper', font_scale= 1.4)
crash_mx = crash_df.corr(numeric_only= True)
sns.heatmap(crash_mx, annot= True, cmap= 'Blues')
plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------------
# Heatmaps-2:
plt.figure(figsize = (10,5))
flights = flights.pivot_table(index= 'month', columns= 'year', values= 'passengers')
sns.heatmap(flights, cmap= 'Greens', linecolor= 'white', linewidths= 1)
plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------------
# Cluster Map-1:
species = iris.pop('species')
sns.clustermap(iris)
plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------------
# Cluster Map-2:
sns.clustermap(flights, cmap= 'Blues', standard_scale= 1)
plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------------
# Pair Grid-1:
iris_g = sns.PairGrid(iris, hue= 'species')
iris_g.map_diag(plt.hist)
#iris_g.map_offdiag(plt.scatter)
iris_g.map_upper(plt.scatter)
iris_g.map_lower(sns.kdeplot)
plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------------
# Pair Grid-2:
iris_g = sns.PairGrid(iris, hue= 'species', x_vars= ["sepal_length", "sepal_width"], y_vars= ["petal_length", "petal_width"])
iris_g.map(plt.scatter)
iris_g.add_legend()
plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------------
# Facet Grid-1:
tips_fg = sns.FacetGrid(tips_df, col= 'time', row= 'smoker')
tips_fg.map(plt.hist, 'total_bill', bins= 8)
plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------------
# Facet Grid-2:
tips_fg = sns.FacetGrid(tips_df, col= 'time', row= 'smoker')
tips_fg.map(plt.scatter, 'total_bill', 'tip')
plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------------
# Facet Grid-3:
tips_fg = sns.FacetGrid(tips_df, col= 'time', hue= 'smoker', height = 4, aspect = 1.3, col_order= ['Dinner', 'Lunch'], palette= 'Set1')
tips_fg.map(plt.scatter, 'total_bill', 'tip')
plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------------
# Facet Grid-4:
kws = dict(s= 50, linewidth= .5, edgecolors= 'w')
tips_fg = sns.FacetGrid(tips_df, col= 'sex', hue= 'smoker', height = 4, aspect = 1.3, hue_order= ['Yes', 'No'], hue_kws= dict(marker= ['^', 'v']))
tips_fg.map(plt.scatter, 'total_bill', 'tip', **kws)
plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------------
# Facet Grid-5:
att_fg = sns.FacetGrid(att_df, col= 'subject', col_wrap= 5, height = 1.5)
att_fg.map(plt.plot, 'solutions', 'score', marker= '.')
plt.show()
