import numpy as np
#----------------------------------------------------
a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
print ("Array a:")
print (a)
print ("------------------------------")
print ("Change a specific element:")
a[1,5] = 20
print (a)
print ("------------------------------")
print ("Change a specific row:")
a[0,:] = 5
print (a)
print ("------------------------------")
print ("Change a specific column:")
a[:,2] = [1,2]
print (a)
#----------------------------------------------------
b = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print ("------------------------------\nArray b:")
print (b)
print ("------------------------------")
print ("Change a specific part:")
b[:,1,:] = [[9,9],[8,8]]
print (b)
