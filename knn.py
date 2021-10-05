import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


# make the dataset with make_blobs use random state 0 use 300 samples
# And plot it
X,Y = make_blobs(n_samples=300,centers=4,cluster_std=0.6,random_state=0)
plt.scatter(X[:,0], X[:,1], c= Y)
plt.title('Plotting a Sample set')
X[0], X[1]


# implement square diff
def square_diff(v1, v2):
    arr= np.array([(v1[0]-v2[0])**2, (v1[1]-v2[1])**2])
    return abs(arr)


# implement root sum squares
def root_sum_squared(v1):
    sq= np.sqrt(v1[0]+v1[1])
    return sq
root_sum_squared(square_diff(X[0],X[1]))


# creating KNN function
def euclidean_distances(v1,v2):
    dist = np.linalg.norm(v1 - v2)
    return dist


#implement the evaluate function RETURN THE A VALUE BETWEEN 0 AND 1
#This cell will be evaluated later on
def evaluate(y, y_p):
    arr= np.array([y == y_p])
    return (np.count_nonzero(arr == True)/len(y))

# evaluate(a,b)



# create KNN prediction function

#Implement the KNN function that predicts the class for the test values using the train values

#OUTPUT MUST BE A NP ARRAY
from sklearn.neighbors import KNeighborsClassifier
#from knn_scratch import KNN
from sklearn.model_selection import train_test_split


def predict(x_test, x_true, y_true, k= 5):
    # YOUR CODE HERE
    knn = KNeighborsClassifier(n_neighbors= k)
    knn.fit(x_true, y_true)
    y_pred= knn.predict(x_test)
    return y_pred


from sklearn.model_selection import train_test_split
#tested with random state 0
#create the train test split test_size 0.2
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# predictions
predictions = predict(x_test,x_train, y_train, k=3)

# predictions = predict(x_test,x_train, y_train, k=2)

print('Accuracy {:0.2f}%'.format(evaluate(predictions, y_test)*100))