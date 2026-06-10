import numpy as np
#--------------------------------
sales = np.array([100,120,90,150,180,110,130])
print ("Sales:", sales)
print ("------------------------------")
#Get the sum:
print ("Total Sales  = :", np.sum(sales))
#Get the mean:
print("Average Sales = ", np.mean(sales))
#Get the maximum:
print("Max Sales     = ", np.max(sales))
#Get the minimum:
print("Min Sales     = ", np.min(sales))
#Show sales above 120:
print("High Sales    = ", (sales[sales > 120]))
