import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
gas = pd.read_csv('gas_prices.csv')
plt.figure(figsize = (5,4), dpi = 150)
plt.title('Gas Prices over Time (in USD)', fontdict={'fontweight': 'bold','fontsize':18})
plt.plot(gas.Year, gas.USA, '.-b', label = 'United States')
plt.plot(gas.Year, gas.Canada, '.-r', label = 'Canada')
plt.plot(gas.Year, gas['South Korea'], '.-g', label = 'South Korea')
plt.plot(gas.Year, gas.Australia, '.-y', label = 'Australia')
# Another way to plot all countries:
# for country in gas:
#     if country != 'Year':
#         plt.plot(gas.Year, gas[country], label = country, marker = '.')
plt.xticks(gas.Year[::3].tolist()+[2011])
plt.xlabel('Year')
plt.ylabel('US Dollars per Gallon')
plt.legend()
plt.savefig('03-Gas Price Figure.png', dpi = 300)
plt.show()
