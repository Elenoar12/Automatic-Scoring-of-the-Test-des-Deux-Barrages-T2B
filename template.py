import numpy as np
import random
from scipy.sparse import csr_matrix
x = [1,1,1,1,1,1,
     2,2,2,2,2,2,
     3,3,3,3,3,3,3,3,3,3,
     4,4,4,4,4,4,
     5,5,5,5,5,5,5,5,
     6,6,6,6,6,
     7,7,7,7,7,7,
     8,8,8,8,
     9,9,9,9,9,9,
     10,10,10,10,10,
     11,11,11,11,11,11,11,11,
     12,12,12,12,12,12,
     13,13,13,13,13,
     14,14,14,14,14,14,
     15,15,15,15,
     16,16,16,16,16,
     17,17,17,17,17,17,
     18,18,18,18,18,18,
     19,19,19,19,19,19,19,
     20,20,20,20,20,20,20,
     21,21,21,21,21,21,21,
     22,22,22,22,22,22,22,22,
     23,23,23,23,
     24,24,24,24,24,
     25,25,25,25,25,25,25,25,
     26,26,26,26,26,26,26,26,
     27,27,27,27,27,27,
     28,28,28,
     29,29,29,29,29,29,29,29,29,
     30,30,30,30,
     31,31,31,31,31,31,31,31,
     32,32,32,32,32,32,32,32,
     33,33,33,33,33,33,33,
     34,34,34,34,
     35,35,35,35,35,
     36,36,36,
     37,37,37,37,37,37,37,
     38,38,38,38,38,38,38,
     39,39,39,39,39,39,39,39]      #rows
y = [1,2,5,9,12,19,
     3,6,10,13,16,23,
     0,1,3,4,5,9,18,20,23,24,
     1,8,11,15,18,21,
     0,4,8,12,16,19,20,24,
     1,8,11,15,19,
     2,6,12,14,17,22,
     2,3,6,11,
     3,8,9,13,18,23,
     0,7,10,14,24,
     2,4,5,9,13,15,17,18,
     2,5,10,14,16,22,
     8,12,16,20,22,
     3,6,11,21,23,24,
     3,10,17,23,
     5,9,11,14,18,
     1,6,10,13,16,21,
     1,6,12,17,21,22,
     0,4,7,15,19,20,24,
     1,6,9,13,16,17,21,
     0,4,13,14,17,20,24,
     2,8,11,12,14,16,20,23,
     1,6,15,19,
     1,7,11,18,21,
     3,7,10,14,15,19,23,24,
     0,4,5,9,15,19,21,22,
     2,5,6,9,17,22,
     7,13,19,
     0,4,7,10,14,15,17,20,24,
     3,5,16,23,
     0,8,11,14,18,20,21,22,
     3,4,7,10,12,17,20,23,
     2,7,8,12,16,18,22,
     6,10,14,21,
     1,11,12,18,21,
     2,8,12,
     3,5,7,11,16,19,23,
     0,4,7,13,18,22,24,
     2,8,9,10,13,15,19,22]      #columns
def listmaker(n):
    listofones = [1] * n
    return listofones
value = listmaker(241)        #list of ones for each true positive position
row = np.array(x)
col = np.array(y)
data = np.array(value)
template = csr_matrix((data, (row, col)), shape=(40, 25)).toarray()
print(template)

#sample_size = 241
#random_value = [random.randint(0,1) for _ in range(sample_size)]      #random values of 0 or 1 in position of true positive
#random_data = np.array(random_value)
#random_trx = csr_matrix((random_data, (row, col)), shape=(40, 25)).toarray()
#print(random_trx)

from test import test

# we find the predicted and true labels that are assigned to some specific class
# then we use the "AND" operator to combine the results of the two label vectors
# into a single binary vector
# then we sum over the binary vector to count how many incidences there are

# True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
TP = np.sum(np.logical_and(test == 1, template == 1))

# True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
TN = np.sum(np.logical_and(test == 0, template == 0))

# False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
FP = np.sum(np.logical_and(test == 1, template == 0))

# False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
FN = np.sum(np.logical_and(test == 0, template == 1))

print(TP)
print(TN)
print(FP)
print(FN)

# Accuracy (ACC): accuracy is the fraction of predictions our model got right
ACC = (TP + TN) / (TP + TN + FP + FN)
print(ACC)

FN_loc = np.logical_and(test == 0, template == 1)
FN_indx = np.transpose(np.nonzero(FN_loc == True))
print(FN_indx)

FP_loc = np.logical_and(test == 1, template == 0)
FP_indx = np.transpose(np.nonzero(FP_loc == True))
print(FP_indx)








