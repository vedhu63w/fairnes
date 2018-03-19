from numpy 					import argsort, cov, column_stack, dot
from numpy.random 			import normal, randn, rand, uniform
from numpy.linalg			import eig
from sklearn.decomposition	import PCA


def generate_data():
	N = 1000 
	X_1 = uniform(0, 15, N)
	X_2 = X_1 * 2
	X_3 = normal(3, 2, N)
	X = column_stack((X_1, X_2, X_3))

	Y = X_1*2 + X_3*3 + randn(N)
	return X, Y

X, Y = generate_data()

N, num_features = X.shape
sensitive_column = 0				#Define the sensitive column

covar = cov(X, rowvar=False)
eigenval, eigenvec = eig(covar)
print eigenvec
sorttosensattr = argsort(eigenvec[sensitive_column,:])
print sorttosensattr

for i in range(1, num_features+1):
	loc_eig = eigenvec[:,sorttosensattr[:i]]
	X_transform = dot(X,loc_eig)

from sklearn.decomposition  import PCA
pca = PCA(n_components=num_features)
X_trans = pca.fit_transform(X)
print dot(X, eigenvec)
print X_trans
import pdb
pdb.set_trace()
# TODO: Add default as well 