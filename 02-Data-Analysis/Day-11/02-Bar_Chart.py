import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
labels = ['A','B','C']
values = [1,4,2]
bars = plt.bar(labels, values)
bars[0].set_hatch('/')
bars[1].set_hatch('O')
bars[2].set_hatch('*')
# It can also be written like:
# patterns = ['/','O','*']
# for bar in bars:
#
plt.savefig('02-Bar Chart.png', dpi = 300)
plt.show()
