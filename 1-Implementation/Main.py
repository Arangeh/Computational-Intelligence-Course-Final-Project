import csv
from RBF import *
import random
import matplotlib.pyplot as plt
from collections import Counter
import itertools

filename = '2clstrain1200.csv'
# filename = '5clstrain1500.csv'
# filename = '4clstrain1200.csv'
train_percentage = 70
test_percentage = 30
reader = csv.reader(open(filename, 'r'), delimiter=',')
lst = []


def countDistinct(arr):
	# counter method gives dictionary of elements in list
	# with their corresponding frequency.
	# using keys() method of dictionary data structure
	# we can count distinct values in array
	return len(Counter(arr).keys())


for row in reader:
	row = list(map(float, row))
	if len(row) == 0:
		continue
	row[len(row) - 1] = int(row[len(row) - 1])
	lst.append(row)

global n_dim
NData = np.shape(lst)[0]

n_dim = np.shape(lst)[1] - 1  # number of dimensions for each sample point from dataset
'''
Note that X is a NData * n_dim array
lst is a NData * (n_dim + 1) array in which an extra column is allocated for storing the class labels for dataset
It has one extra column more than that of X
'''
random.shuffle(lst)
# X_Test = []

N_Train = int((train_percentage / 100) * NData)
N_Test = NData - N_Train

X_Train = [item[0:n_dim] for item in lst[0:N_Train]]
Y_Train = [item[n_dim] for item in lst[0:N_Train]]
Y_Total = [item[n_dim] for item in lst[0:NData]]  # for counting the number of clusters
X_Total = [item[0:n_dim] for item in lst[0:NData]]  # used in creating the sample

X_Test = [item[0:n_dim] for item in lst[N_Train:NData]]
Y_Test = [item[n_dim] for item in lst[N_Train:NData]]


def create_sample(X_min, X_max, N):
	'''
	for each feature
	:param X_min:
	:param X_max:
	:param N: determines the number of intervals having equal length
	 in dividing the interval between X_min_i and X_max_i
	:return: an ndarray, containing all the sample points
	it has  (N + 1) ** n_dim elements
	'''
	print(X_min)
	print(X_max)
	n = len(X_min)
	dimension = list(np.zeros(n))
	for i in range(n):
		delta = (X_max[i] - X_min[i]) / N
		dimension[i] = [(X_min[i] + delta * j) for j in range(N + 1)]
	cart_prod = itertools.product(*dimension)
	res = []
	for e in cart_prod:
		e = np.asarray(e)
		res.append(e)
	res = np.asarray(res, float)
	return res


if __name__ == '__main__':
	
	# First, we run FCM to compute the centers we need for training the RBF
	step = 2
	
	par.X = X_Train
	par.Y = Y_Train
	par.Y_Test = Y_Test
	par.classNum = countDistinct(Y_Total)
	
	min_clusters = 1
	max_clusters = 4
	CLUSTER = list(range(min_clusters, max_clusters, step))
	ACCUR = []
	# '''
	for i in range(min_clusters, max_clusters, step):
		par.c = i
		fcm = FCM(X_Train, par.fuzziness, par.c, par.epsilon)
		fcm.run()
		rbf = RBF(X_Train, par.Y, par.V, par.gamma)
		rbf.train()
		rbf.test(X_Test)
		print("number of clusters = ", par.c)
		print("RBF Train Accuracy = ", par.RBF_train_accuracy)
		print("RBF Test Accuracy = ", par.RBF_test_accuracy)
		ACCUR.append(par.RBF_test_accuracy)
	f = plt.figure(1)
	plt.ylabel('Classification Accuracy')
	plt.xlabel('Number of Clusters')
	plt.plot(CLUSTER, ACCUR)
	plt.show()
	# '''
	
	# '''
	# For using RBF with dataset ‘5clstrain1500.csv’, uncomment the following 2 lines
	par.c_optimized_1500_5 = 27
	par.c = par.c_optimized_1500_5
	
	# For using RBF with dataset ‘4clstrain1200.csv’, uncomment the following 2 lines
	# par.c_optimized_1200_4 = 25
	# par.c = par.c_optimized_1200_4
	
	# For using RBF with dataset ‘2clstrain1200.csv’, uncomment the following 2 lines
	# par.c_optimized_1200_2 = 7
	# par.c = par.c_optimized_1200_2
	
	fcm = FCM(X_Train, par.fuzziness, par.c, par.epsilon)
	fcm.run()
	rbf = RBF(X_Train, par.Y, par.V, par.gamma)
	rbf.train()
	rbf.test(X_Test)
	
	print("number of clusters = ", par.c)
	print("RBF Train Accuracy = ", par.RBF_train_accuracy)
	print("RBF Test Accuracy = ", par.RBF_test_accuracy)
	# The following code draws the plot of Fuzzy Cmeans Clustering result
	# Uncomment to see the plot
	
	f = plt.figure(2)
	plt.ylabel('Y')
	plt.xlabel('X')
	plt.scatter([point[0] for point in X_Train], [point[1] for point in X_Train])
	V = par.V
	for i in range(par.c):
		plt.scatter(V[i][0], V[i][1])
	plt.show()
	f = plt.figure(3)
	
	# Create sample for visualizing the clusters' boundaries
	X_Total = np.asarray(X_Total, float)
	# X_min is a 1 * n_dim vector whose i-th component is the
	# minimum of i-th component amongst all the vectors X in X_Total
	X_min = [np.min(X_Total[:, i]) for i in range(n_dim)]
	# X_min is a 1 * n_dim vector whose i-th component is the
	# minimum of i-th component amongst all the vectors X in X_Total
	X_max = [np.max(X_Total[:, i]) for i in range(n_dim)]
	plt.figure(3)
	sample = create_sample(X_min, X_max, 99)
	# print(np.shape(sample))
	U = FCM.calculate_membership(sample, par.V)
	
	clusters = np.zeros((par.c,), dtype=np.ndarray)
	
	NSamples = np.shape(sample)[0]
	
	for i in range(par.c):
		clusters[i] = []
	for j in range(NSamples):
		label_j = np.argmax(U[j, :])
		clusters[label_j].append(sample[j, :])
	
	for i in range(par.c):
		clusters[i] = np.asarray(clusters[i])
		plt.scatter(clusters[i][:, 0], clusters[i][:, 1], s=1)
	plt.show()
	
	Y_predicted = par.Y_predicted
	Y_Test = par.Y_Test
	
	classNum = par.classNum
	X_Test = np.asarray(X_Test, float)
	classes = np.zeros((classNum,), dtype=np.ndarray)
	f = plt.figure(4)
	
	for i in range(classNum):
		classes[i] = []
	for i in range(np.shape(Y_Test)[0]):
		if (Y_Test[i] == Y_predicted[i]):
			classes[RBF.map(Y_Test[i])].append(X_Test[i, :])
		else:
			plt.scatter(X_Test[i][0], X_Test[i][1], c='r')
	
	for i in range(classNum):
		classes[i] = np.asarray(classes[i])
		plt.scatter(classes[i][:, 0], classes[i][:, 1], s=1)
	plt.show()
# '''
