import numpy as np
#--------------------------------------
a = np.array([1,2,3])
print ("a:", a)
print ("------------------------------")
b = a.copy()
b[0] = 100
print ("b = a.copy():")
print ("b:", b)
print ("a:", a)
print ("------------------------------")
c = a
c[0] = 100
print ("c = a")
print ("c:", c)
print ("a:", a)