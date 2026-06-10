import numpy as np
#--------------------------------
temps = np.array([25,28,31,22,35,30,27])
print ("Temperatures:", temps)
print ("------------------------------")
#Get the mean:
print("Average Temperature = ", np.mean(temps))
#Get the maximum:
print("Max Temperature     = ", np.max(temps))
#Get the minimum:
print("Min Temperature     = ", np.min(temps))
#Show temperatures above 30:
print("High Temperature    = ", (temps[temps > 30]))
