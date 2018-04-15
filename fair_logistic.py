import numpy as np
import math
import load_compass as data_loader
import load_adult_data as adult_data_loader
import load_synthetic_data as synthetic_data_generator

alpha = 15.0		# alpha is the fairness multiplicative constant for y_bar
beta = 15.0			# beta is the fairness multiplicative constant for y
lr = 0.35		# learning rate
training_iterations = 400
np.set_printoptions(precision=5)
def h(theta_x):
	# Sigmoid of the value
	sigmoid = 1.0 / (1.0 + np.exp(-theta_x))
	if sigmoid == 0.0:
		sigmoid = 1e-100
	if sigmoid == 1.0:
		sigmoid = 9.99
	return sigmoid

def test_fair_logistic(X, Y, s_id, theta):
	n, d = X.shape

	c_s1_y_bar = 0.0
	c_s2_y_bar = 0.0
	c_s1_y = 0.0
	c_s2_y = 0.0
	for i in range(n):
		if Y[i] == 0:
			if X[i, s_id] == 0:
				c_s1_y_bar += 1.0
			elif X[i, s_id] == 1:
				c_s2_y_bar += 1.0
		elif Y[i] == 1:
			if X[i, s_id] == 0:
				c_s1_y += 1.0
			elif X[i, s_id] == 1:
				c_s2_y += 1.0

	delta_y_bar_root = 0.0
	delta_y_bar_true = 0.0
	delta_y_root = 0.0
	delta_y_true = 0.0
	for i in range(n):
		x_i = X[i,:]
		theta_x = theta.dot(x_i)
		h_x_i = h(theta_x)

		y_cap = 1 if h_x_i >= 0.5 else 0

		if(Y[i] == 0):
			if(x_i[s_id] == 0):
				# s1 category
				delta_y_bar_root += h_x_i / c_s1_y_bar
				delta_y_bar_true += y_cap / c_s1_y_bar
			elif(x_i[s_id] == 1):
				# s2 category
				delta_y_bar_root -= h_x_i / c_s2_y_bar
				delta_y_bar_true -= y_cap / c_s2_y_bar
		elif(Y[i] == 1):
			if(x_i[s_id] == 0):
				# s1 category
				delta_y_root -= h_x_i / c_s1_y
				delta_y_true -= y_cap / c_s1_y
			elif(x_i[s_id] == 1):
				# s2 category
				delta_y_root += h_x_i / c_s2_y
				delta_y_true += y_cap / c_s2_y
	delta_y_bar_root = abs(delta_y_bar_root)
	delta_y_root = abs(delta_y_root)
	accuracy = get_accuracy(X, Y, theta)
	print "Test:", delta_y_bar_root, "\t", delta_y_root, "\t", abs(delta_y_bar_true), "\t", abs(delta_y_true), "\t", accuracy
	return delta_y_bar_root, delta_y_root, accuracy

def get_accuracy(X, Y, theta):
	correct = 0.0
	n = X.shape[0]
	for i in range(n):
		x_i = X[i,:]
		theta_x = theta.dot(x_i)
		h_x_i = h(theta_x)
		y_cap = 1 if h_x_i >= 0.5 else 0
		
		if y_cap == Y[i]:
			correct += 1.0
	return correct / float(n)

def train_fair_logistic(X, Y, s_id, X_test, Y_test):
	global lr
	# We get 3 inputs
	# 1 - X = n feature vectors each of d dimension
	# 2 - Y = n labels corresponding to the features
	# 3 - s_id = column id of the sensitive feature
	# We have to train the fair_logistic loss using full gradient descent

	n, d = X.shape
	import time
	np.random.seed(int(time.time() * 100000)%100000)
	theta = np.random.randn(d)
	print theta

	c_s1_y_bar = 0.0
	c_s2_y_bar = 0.0
	c_s1_y = 0.0
	c_s2_y = 0.0
	for i in range(n):
		if Y[i] == 0:
			if X[i, s_id] == 0:
				c_s1_y_bar += 1.0
			elif X[i, s_id] == 1:
				c_s2_y_bar += 1.0
		elif Y[i] == 1:
			if X[i, s_id] == 0:
				c_s1_y += 1.0
			elif X[i, s_id] == 1:
				c_s2_y += 1.0

	for j in range(training_iterations):
		# Calculate Likelihood loss gradients
		like_loss_grad = np.zeros(d, dtype=np.float)
		like_loss = 0.0
		for i in range(n):
			x_i = X[i,:]
			theta_x = theta.dot(x_i)
			h_x_i = h(theta_x)
			like_loss_grad_i = x_i * (Y[i] - h_x_i)
			like_loss_grad += like_loss_grad_i
			like_loss += Y[i] * np.log(h_x_i) + (1-Y[i]) * np.log(1 - h_x_i)
		like_loss_grad /= float(n)
		like_loss /= float(n)
		
		# Calculate Fairness constraint gradients (delta) for y_bar
		first_term_numerator_y_bar = np.zeros(d, dtype=np.float)
		second_term_numerator_y_bar = np.zeros(d, dtype=np.float)
		first_term_numerator_y = np.zeros(d, dtype=np.float)
		second_term_numerator_y = np.zeros(d, dtype=np.float)
		delta_y_bar_root = 0.0
		delta_y_bar_true = 0.0
		delta_y_root = 0.0
		delta_y_true = 0.0
		for i in range(n):
			x_i = X[i,:]
			theta_x = theta.dot(x_i)
			h_x_i = h(theta_x)

			y_cap = 1 if h_x_i >= 0.5 else 0

			if(Y[i] == 0):
				if(x_i[s_id] == 0):
					# s1 category
					first_term_numerator_y_bar += (h_x_i * (1 - h_x_i)) * x_i
					delta_y_bar_root += h_x_i / c_s1_y_bar
					delta_y_bar_true += y_cap / c_s1_y_bar
				elif(x_i[s_id] == 1):
					# s2 category
					second_term_numerator_y_bar += (h_x_i * (1 - h_x_i)) * x_i
					delta_y_bar_root -= h_x_i / c_s2_y_bar
					delta_y_bar_true -= y_cap / c_s2_y_bar
			elif(Y[i] == 1):
				if(x_i[s_id] == 0):
					# s1 category
					first_term_numerator_y += (h_x_i * (1 - h_x_i)) * x_i
					delta_y_root -= h_x_i / c_s1_y
					delta_y_true -= y_cap / c_s1_y
				elif(x_i[s_id] == 1):
					# s2 category
					second_term_numerator_y += (h_x_i * (1 - h_x_i)) * x_i
					delta_y_root += h_x_i / c_s2_y
					delta_y_true += y_cap / c_s2_y
		first_term_y_bar = first_term_numerator_y_bar / c_s1_y_bar
		second_term_y_bar = second_term_numerator_y_bar / c_s2_y_bar

		first_term_y = first_term_numerator_y / c_s1_y
		second_term_y = second_term_numerator_y / c_s2_y

		delta_y_bar_grad = 2.0 * delta_y_bar_root * (first_term_y_bar - second_term_y_bar)
		delta_y_grad = 2.0 * delta_y_root * (second_term_y - first_term_y)

		# make roots absolute
		delta_y_bar_root = abs(delta_y_bar_root)
		delta_y_root = abs(delta_y_root)

		# Total gradients
		full_grad = -1.0 * like_loss_grad + alpha * delta_y_bar_grad + beta * delta_y_grad

		# Updating the parameters
		
		theta = theta - lr * full_grad

		# Compute accuracy
		accuracy = get_accuracy(X, Y, theta)
		# if (j+1)%50 == 0:
		# 	lr /= 2.0
		if j==0 or (j+1)%25 == 0:
			# print (j+1), "\t", delta_root ** 2, "\t", correct / n
			total_loss = -1.0* like_loss + alpha * delta_y_bar_root ** 2 + beta * delta_y_root ** 2
			print (j+1), "\t", abs(delta_y_bar_root), "\t", abs(delta_y_root), "\t", -1.0 * like_loss, "\t", total_loss, "\t", accuracy
			# print (j+1), "\t", like_loss_new, "\t", correct / n
			# print (j+1), "\t", -1.0 * like_loss, "\t", correct / n
			# test_fair_logistic(X_test, Y_test, s_id, theta)
	# Retrun the learned weights
	print alpha
	return theta

"""
X, y, x_control = data_loader.load_compas_data()
s_id = 4
x_control = np.array(x_control['race']).reshape((len(x_control['race']), 1))
print np.sum(y)
print np.sum(x_control)
print x_control.shape
X_full = X
"""

X, y, x_sensitive = adult_data_loader.load_adult_data()
s_id = 9
X, y, x_control = synthetic_data_generator.generate_synthetic_data(1,1000)
s_id = 3
# Add x_control and intercept to X
# intercept = np.array([[1.0] for _ in range(X.shape[0])])
intercept_and_x_control = np.array([[1.0, x] for x in x_control])
print intercept_and_x_control.shape
X_full = np.append(X, intercept_and_x_control, axis = 1)
print X_full.shape
# print X[0:4]
print y.shape
y[y==-1] = 0


# s_id = 4

indices = np.random.permutation(X_full.shape[0])
split_id = int(indices.shape[0] * 0.9)

X_train = X_full[indices[:split_id],:]
y_train = y[indices[:split_id]]
X_test = X_full[indices[split_id:],:]
y_test = y[indices[split_id:]]

theta = train_fair_logistic(X_train, y_train, s_id, X_test, y_test)
delta_y_bar_root, delta_y_root, accuracy = test_fair_logistic(X_test, y_test, s_id, theta)
print theta