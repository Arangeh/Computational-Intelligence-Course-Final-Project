import parameters as par
import numpy as np


class FCM:
	def __init__(self, X, m, c, epsilon):
		global U
		self.X = np.asarray(X, dtype=np.float)
		self.m = m
		self.c = c
		self.epsilon = epsilon
		
		'''
		Considering X as an NData * n_dim matrix, we should create a membership matrix U
		whose dimension is NData * c, where c is the number of clusters, initialized by ranodom numbers which satisfy
		the necessary conditions of membership function.
		'''
		NData = np.shape(X)[0]
		# Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1)
		self.U = np.random.rand(NData, c)
		for i in range(NData):
			self.U[i][:] = self.U[i][:] / np.sum(self.U[i][:])
	
	def run(self):
		# Having initial U we compute the centers
		# '''
		while True:
			self.compute_cluster_centers()
			self.update_membership()
			if self.check_termination_criterion(self.U, self.Uold):
				break
			# '''
		# After computing the centers we compute
		# Final V and U matrices will be stored in parameters class
		par.V = self.V
		par.U = self.U
		return
	
	def update_membership(self):
		# First we compute the distance matrix whose dimensions are the same as U
		# distance here is the Euclidean distance
		self.Uold = np.array(self.U)
		self.U = self.calculate_membership(self.X, self.V)
		return
	
	@staticmethod
	def calculate_membership(X, V):
		Num_of_X = np.shape(X)[0]  # number of sample points
		Num_of_Clusters = np.shape(V)[0]  # number of cluster centers, or simply the number of clusters
		
		D = np.zeros((Num_of_X, Num_of_Clusters))  # distance matrix
		
		for i in range(Num_of_X):
			for j in range(Num_of_Clusters):
				D[i][j] = (np.linalg.norm(X[i][:] - V[j][:]))
		m = par.fuzziness
		D = np.power(D, (2 / (m - 1)))  # elementwise power function for the ndarray matrix in numpy
		D = np.reciprocal(D)  # elementwise 1/x for the matrix. Helps in the calculation of updated memberships
		
		# Now, we can compute the updated membership matrix
		
		# Note that U is the same dimension as D
		U = np.zeros((Num_of_X, Num_of_Clusters))
		for k in range(Num_of_X):
			for i in range(Num_of_Clusters):
				U[k][i] = (D[k][i]) / (np.sum(D[k][:]))
		
		return U
	
	def compute_cluster_centers(self):
		UT = np.transpose(self.U)
		m = par.fuzziness
		UT = np.power(UT, m)
		self.V = np.dot(UT, self.X)
		# Now, we should normalize V. Note that V has a c * n_dim dimension, where
		for i in range(par.c):
			self.V[i][:] = self.V[i][:] / np.sum(UT[i][:])
		return
	
	def check_termination_criterion(self, U, Uold):
		'''
		:param U: ndarray current U
		:param Uold: ndarray previous U
		:return: returns true if |U - Uold| is less than epsilon, false otherwise
		'''
		res = False
		if (np.linalg.norm(U - Uold) < par.epsilon):
			res = True
		return res
