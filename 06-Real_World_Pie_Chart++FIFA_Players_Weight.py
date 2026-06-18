import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
fifa = pd.read_csv('fifa_data.csv')
fifa.Weight = [int(x.strip('lbs')) if type(x) == str else x for x in fifa.Weight]
light = (fifa['Weight'] < 125).sum()
light_medium = ((125 < fifa['Weight']) & (fifa['Weight'] < 150)).sum()
medium = ((150 < fifa['Weight']) & (fifa['Weight'] < 175)).sum()
medium_heavy = ((175 < fifa['Weight']) & (fifa['Weight'] < 200)).sum()
heavy = (200 < fifa['Weight']).sum()
weights = [light, light_medium, medium, medium_heavy, heavy]
labels = ['-125','125-150','150-175','175-200','+200']
plt.style.use('ggplot')
explode = (.4,.2,0,0,.4)
plt.title('Weight Distribution of FIFA Players (in lbs)')
plt.pie(weights, labels = labels, autopct='(%.2f %%)', pctdistance= .75, explode= explode)
plt.savefig('06-FIFA 2018 Weight Distribution Pie Chart.png', dpi = 300)
plt.show()
