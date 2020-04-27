from FCM import *
from numpy.linalg import inv

global G  # interpolation matrix
global predicted  # predicted values
global W  # weights
global diff
global U
global C
global Y_EXPANDED
global accuracy


class RBF:
	def __init__(self, X, Y, V, gamma):
		global U
		self.X = np.asarray(X, dtype=np.float)  # here X is used for training
		self.Y = np.asarray(Y, dtype=np.float)
		self.V = V  # centers that are used in calculation of Radial Based Functions
		# (here we have an interpolation matrix named G)
		self.gamma = gamma
		self.calculate_diff()
		U = par.U
		c = np.shape(self.V)[0]
	
	def test(self, X):
		global G
		global U
		global accuracy
		global predicted
		
		self.X = np.asarray(X, dtype=np.float)  # here X is used for testing
		self.Y = np.asarray(par.Y_Test, dtype=np.float)  # here Y is used for testing
		
		U = FCM.calculate_membership(self.X, self.V)
		self.calculate_diff()
		self.calculate_covariance_matrix()
		self.calculate_interpolation_matrix()
		self.predict()
		par.RBF_test_accuracy = accuracy
		par.Y_predicted = predicted
	
	def train(self):
		global G
		global accuracy
		self.calculate_diff()
		self.calculate_covariance_matrix()
		self.calculate_interpolation_matrix()
		self.calculate_Y_EXPANDED()
		self.calculate_weights()
		self.predict()
		par.RBF_train_accuracy = accuracy
		return
	
	def calculate_diff(self):
		global diff
		X = self.X
		V = self.V
		NData = np.shape(X)[0]
		c = np.shape(V)[0]  # number of clusters
		
		diff = np.zeros((NData, c), dtype=object)
		# Initializing the matrix of matrices
		for k in range(NData):
			for j in range(c):
				X_k = X[k][:]
				V_j = V[j][:]
				deltaXV = X_k - V_j
				'''
				to produce an n * n matrix from production of an n * 1 matrix
				by a 1 * n matrix, we have used 'np.outer'
				here, it calculates the outer product of two vectors
				which gives us the desired result
				'''
				diff[k][j] = np.outer(np.transpose(deltaXV), deltaXV)  # an n * n matrix
		return
	
	def calculate_covariance_matrix(self):
		'''
		:param i: index of vector in V
		:return: returns the covariance matrix, defined according to the formula that is stated in the project definition
		note that X_k and V_i are of the same dimension of n * 1
		Also note that covariance matrix is an n * n matrix where n is the dimension of vector X_k and V_i
		'''
		global U
		global diff
		global C
		
		m = par.fuzziness
		c = np.shape(self.V)[0]
		C = np.zeros((c,), dtype=object)
		UT = np.transpose(U)
		UT = np.power(UT, m)
		for i in range(c):
			C[i] = np.dot(UT[i, :], diff[:, i])
			# Now, we should normalize C[i]
			C[i] = C[i] / np.sum(UT[i][:])
		return
	
	def calculate_interpolation_matrix(self):
		'''
		:param X: X can be either the train dataset or the test dataset
		:return: returns the interpolation matrix named G
		'''
		global C
		global G
		X = self.X
		V = self.V
		NData = np.shape(self.X)[0]
		c = np.shape(self.V)[0]
		G = np.zeros((NData, c))
		for k in range(NData):
			for i in range(c):
				X_k = X[k, :]
				V_i = V[i, :]
				deltaXV = X_k - V_i
				G[k][i] = np.exp(-self.gamma * (np.dot(
					np.dot(deltaXV, inv(C[i])), np.transpose(deltaXV))))
		return
	
	def calculate_Y_EXPANDED(self):
		'''
		Note that Y is the label set used for training,
		while Y_EXPANDED is an expanded form of Y, as stated in the project definition
		:return:
		'''
		global Y_EXPANDED
		NData = np.shape(self.X)[0]
		classNum = par.classNum
		Y_EXPANDED = np.zeros((NData, classNum))
		for i in range(NData):
			Y_EXPANDED[i][self.map(int(self.Y[i]))] = 1
		return
	
	def calculate_weights(self):
		global W
		global G
		global Y_EXPANDED
		GT = np.transpose(G)
		W = np.dot(np.dot(inv(np.dot(GT, G)), GT), Y_EXPANDED)
		return
	
	def predict(self):
		global G
		global W
		global predicted
		prediction_matrix = np.dot(G, W)
		NData = np.shape(prediction_matrix)[0]
		predicted = np.zeros((NData,))
		
		for i in range(NData):
			predicted[i] = self.unmap((np.argmax(prediction_matrix[i, :])))
		
		self.report()
		return
	
	def report(self):
		global accuracy
		NData = len(predicted)
		comparison = predicted - self.Y
		accuracy = 1 - np.sum(np.abs(np.sign(comparison))) / NData
		return
	
	@staticmethod
	def map(a):
		# Version1
		'''
		res = a - 1
		'''
		# Version2
		# '''
		res = int((a + 1) / 2)
		# '''
		# Version3
		return res
	
	@staticmethod
	def unmap(a):
		# Version1
		'''
		res = a + 1
		'''
		# Version2
		# '''
		res = 2 * a - 1
		# '''
		# Version3
		return res
