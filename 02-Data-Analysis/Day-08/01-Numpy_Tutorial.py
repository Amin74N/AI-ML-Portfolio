import numpy as np
#--------------------------------------
a = np.array([1,2,3] , dtype = np.int32)
print("Array a:")
print(a)
#--------------------------------------
b = np.array([[1,2,3],[4,5,6]])
print("Array b:")
print(b)
print("------------------------------")
#Get Dimension:
print ("Array a Dimensions:", a.ndim)
print ("Array b Dimensions:", b.ndim)
print("------------------------------")
#Get Shape:
print ("Array a Shape:", a.shape)
print ("Array a Shape:", b.shape)
print("------------------------------")
#Get Type:
print ("Array a Type:", a.dtype)
print ("Array B Type:", b.dtype)
print("------------------------------")
#Get Size:
print ("Array a Item Size:", a.itemsize)
print ("Array b Item Size:", b.itemsize)
print("------------------------------")
#Get Number of Elements:
print ("Array a Number of Elements:", a.size)
print ("Array b Number of Elements:", b.size)
print("------------------------------")
#Get Total Size:
print ("Array a Total Size:", a.size * a.itemsize)
print ("Array a Total Size:", a.nbytes)
print ("Array b Total Size:", b.size * b.itemsize)
print ("Array b Total Size:", b.nbytes)
