global X  # dataset
global U  # membership matrix
global V  # containing means(cluster centers)
global epsilon  # used in termination criterion
global fuzziness
global c  # number of clusters
global classNum
global Y_Test
global Y_predicted
# optimized number of clusters that gives us the best
#  accuracy in '5clstrain1500.csv'
global c_optimized_1500_5
# optimized number of clusters that gives us the best
#  accuracy in '4clstrain1200.csv'
global c_optimized_1200_4
# optimized number of clusters that gives us the best
#  accuracy in '2clstrain1200.csv'
global c_optimized_1200_2
global RBF_train_accuracy
global RBF_test_accuracy
fuzziness = 2
epsilon = 0.01
# c = 15
gamma = 0.1
