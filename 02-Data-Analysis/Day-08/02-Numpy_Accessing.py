import numpy as np
#--------------------------------------------------------------
a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
print ("Array a:")
print (a)
#--------------------------------------------------------------
#Get a specific element [r,c]:
print ("Access a specific element:", a[1,5])
#--------------------------------------------------------------
#Get a specific row:
print ("Access a specific row:", a[0,:])
#--------------------------------------------------------------
#Get a specific column:
print ("Access a specific column:", a[:,2])
#--------------------------------------------------------------
#Get a little more fancy [start index : end index : step size]:
print ("A little more fancy:", a[0,1:-1:2])
#--------------------------------------------------------------
b = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print ("-------------------------------------------\nArray b:")
print (b)
#--------------------------------------------------------------
#Get a specific element (work outside in):
print ("Access a specific element:", b[0,1,1])
#--------------------------------------------------------------
#Get a specific part:
print ("Access a specific part:")
print (b[:,1,:])
#--------------------------------------------------------------
#Get a more specific part:
print ("Access a more specific part:")
print (b[:,0,0])
