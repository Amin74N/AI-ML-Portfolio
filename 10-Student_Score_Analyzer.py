import numpy as np
#--------------------------------
scores = np.array([18,19,17,20,15])
print ("Scores:", scores)
print ("------------------------------")
#Get the mean:
print("Average score = ", np.mean(scores))
#Get the maximum:
print("Max score     = ", np.max(scores))
#Get the minimum:
print("Min score     = ", np.min(scores))
#Show scores above 18:
print("Top Scores    = ", (scores[scores > 18]))
