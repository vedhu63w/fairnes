from load_compas_data		import load_compas_data
from GetAccFair             import get_acc_fair

from matplotlib				import pyplot as plt
from numpy 					import argsort, average, cov, column_stack, dot
from numpy.random 			import normal, randn, rand, uniform
from numpy.linalg			import eig
from os 					import makedirs as os_makedir
from os.path				import exists as os_exists, join as os_join
from sklearn.decomposition	import PCA


# def generate_data():
# 	N = 1000 
# 	X_1 = uniform(0, 15, N)
# 	X_2 = X_1 * 2
# 	X_3 = normal(3, 2, N)
# 	X = column_stack((X_1, X_2, X_3))

# 	Y = X_1*2 + X_3*3 + randn(N)
# 	return X, Y

# X, Y = generate_data()


def writable(dir_nm, fl_nm):
	if not os_exists(dir_nm):
		os_makedir(dir_nm)
	return os_join(dir_nm, fl_nm)


X, Y, x_control = load_compas_data()
Y = [0 if y==-1 else y for y in Y]
x_control = x_control.values()[0]                           #.values returns a list of list values
X = column_stack((x_control, X))							#First column is sensitive attribute

split = 0.9
N, num_features = X.shape
sensitive_column = 0										#Define the sensitive column

covar = cov(X, rowvar=False)
eigenval, eigenvec = eig(covar)
sorttosensattr = argsort(abs(eigenvec[sensitive_column,:]))		#Sort the columns according to row 0
X_norm = X - average(X, axis=0)

for i in range(1, num_features+1):
	loc_eig = eigenvec[:,sorttosensattr[:i]]
	X_transform = dot(X_norm,loc_eig)
	X_train, Y_train = X_transform[:int(split*N)], Y[:int(split*N)]
	X_test, Y_test = X_transform[int(split*N):], Y[int(split*N):]
	acc, fair = get_acc_fair((X_train, Y_train), (X_test, Y_test), x_control[int(split*N):])
	
	feat_acc_fair = (i, acc, fair)
	plt.scatter(i, acc, color='r')
	plt.scatter(i, fair, color='b')

X_train, Y_train = X[:int(split*N)], Y[:int(split*N)]
X_test, Y_test = X[int(split*N):], Y[int(split*N):]
print "Using Standard Logistic Regression "
print get_acc_fair((X_train, Y_train), (X_test, Y_test), x_control[int(split*N):])

plt.title("PCA_Logistic")
plt.xlabel("Number of Features")
plt.legend(["Accuracy", "Fairness"])
plt.savefig(writable("./Results", "PCA_Logistic.png"))
plt.close()
# TODO: Add default as well 