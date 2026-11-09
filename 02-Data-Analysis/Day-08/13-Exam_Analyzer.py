import numpy as np
#--------------------------------
scores = np.array([12,18,20,15,9,17,19])
print ("Scores:", scores)
print ("------------------------------")
#Get the mean:
print("Average Score = ", np.mean(scores))
#Get the maximum:
print("Top Score     = ", np.max(scores))
#Show qualified students:
print("Qualified     = ", (scores[scores >= 10]))
#Show disqualified students:
print("Disqualified  = ", (scores[scores < 10]))
