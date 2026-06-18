import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data = pd.read_csv('youtube_data.csv')
view_count = data['view_count']
likes = data['likes']
ratio = data['ratio']
plt.scatter(view_count, likes, c=ratio , cmap='summer', edgecolors='black', linewidths=1, alpha = 0.75)
cbar = plt.colorbar()
cbar.set_label('Like/Dislike Ratio')
plt.xscale('log')
plt.yscale('log')
plt.title('Trending YouTube Videos')
plt.xlabel('View Count')
plt.ylabel('Total Likes')
plt.savefig('09-YouTube Like-Dislike Ratio Scatter Plot.png', dpi = 300)
plt.show()
