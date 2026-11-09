import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
fifa = pd.read_csv('fifa_data.csv')
foot_count = fifa['Preferred Foot'].value_counts()
left = foot_count.get('Left', 0)
right = foot_count.get('Right', 0)
labels = ['Left','Right']
colors = ['green','red']
plt.pie([left, right], labels = labels, colors = colors, autopct='%.2f %%')
plt.title('Foot Preference of FIFA Players')
plt.savefig('05-FIFA 2018 Foot Preference Pie Chart.png', dpi = 300)
plt.show()
