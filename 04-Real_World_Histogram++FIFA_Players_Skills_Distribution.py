import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
fifa = pd.read_csv('fifa_data.csv')
bins = [40,50,60,70,80,90,100]
plt.hist(fifa.Overall, bins = bins, color = '#abcdef')
plt.title('Distribution of Player Skills in FIFA 2018')
plt.xticks(bins)
plt.xlabel('Skill Level')
plt.ylabel('Number of Players')
plt.savefig('04-FIFA 2018 Skill Level Histogram.png', dpi = 300)
plt.show()
