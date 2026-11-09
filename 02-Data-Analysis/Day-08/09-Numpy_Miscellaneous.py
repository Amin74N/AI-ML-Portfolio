import numpy as np
#----------------------------------------
#Load data from file:
filedata = np.genfromtxt('Data.txt', delimiter=',')
print(filedata)
print ("---------------------------------------------------------------------------")
filedata = filedata.astype('int32')
print(filedata)
print ("---------------------------------------------------------------------------")
#Boolean masking & advanced indexing:
print(filedata > 50)
print ("---------------------------------------------------------------------------")
print(~(filedata > 50) & (filedata < 100))
print ("---------------------------------------------------------------------------")
print(filedata[filedata > 50])
print ("---------------------------------------------------------------------------")
print(np.any(filedata > 50 , axis=0))
print ("---------------------------------------------------------------------------")
print(np.all(filedata > 50 , axis=0))
print ("---------------------------------------------------------------------------")
#Indexing with a list:
a = np.array([1,2,3,4,5,6,7,8,9])
print(a[[1,2,8]])
