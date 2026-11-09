import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# --------------------------------------------------------------------------------------------------------------------------------------
# Resize the graph (dpi specifies pixels per inch & when saving probably should use 300 if possible):
plt.figure(figsize = (5,4), dpi = 150)
# --------------------------------------------------------------------------------------------------------------------------------------
# Line 1:
x = [0,1,2,3,4]
y = [0,2,4,6,8]
# Keyword argument notation:
plt.plot(x,y, label = 'Y = 2X', color = 'red', linewidth = 2,linestyle = '-.' , marker = '.', markersize = 10, markeredgecolor = 'blue')
# The shorthand form can also be used:
# fmt = '[marker][line][color]'
# plt.plot(x,y, '.--r', label = 'Y = 2X')
# --------------------------------------------------------------------------------------------------------------------------------------
# Line 2:
# Select interval we want to plot points at:
x2 = np.arange(0,4.5,0.5)
# Plot two parts of the line with tow different shapes & colors:
plt.plot(x2[:5],x2[:5]**2, '.-b', label = 'Y = X^2', linewidth = 2, markersize = 10, markeredgecolor = 'red')
plt.plot(x2[4:],x2[4:]**2, '.--g', label = 'Y = X^2', linewidth = 2, markersize = 10, markeredgecolor = 'yellow')
# --------------------------------------------------------------------------------------------------------------------------------------
# Add a title (specify font parameters with fontdict):
plt.title('First Graph!',fontdict={'fontname': 'Comic Sans MS','fontsize':20})
# X & Y labels:
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
# X & Y axis tickmarks (scale of graph):
plt.xticks([0,1,2,3,4])
plt.yticks([0,2,4,6,8])
# Add a legend:
plt.legend()
# Save figure (dpi 300 is good when saving so graph has high resolution):
plt.savefig('01-First Graph.png', dpi = 300)
# Show plot:
plt.show()
