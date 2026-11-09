import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
fifa = pd.read_csv('fifa_data.csv')
plt.figure(figsize = (5,8))
barcelona = fifa.loc[fifa.Club == 'FC Barcelona']['Overall']
madrid = fifa.loc[fifa.Club == 'Real Madrid']['Overall']
revs = fifa.loc[fifa.Club == 'New England Revolution']['Overall']
plt.style.use('default')
labels = ['FC Barcelona','Real Madrid', 'NE Revolution']
boxes = plt.boxplot([barcelona, madrid, revs], tick_labels = labels, patch_artist = True, medianprops ={'linewidth':2})
for box in boxes['boxes']:
    box.set(color='red', linewidth=2)
    box.set(facecolor='green')
plt.title('Professional Soccer Team Comparison')
plt.ylabel('FIFA Overall Rating')
plt.savefig('07-FIFA 2018 Team Comparison Box Plot.png', dpi = 300)
plt.show()
