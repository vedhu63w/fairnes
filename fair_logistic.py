import numpy as np
import math
import load_compass as data_loader

alpha = 0.1		# alpha is the fairness multiplicative constant
lr = 0.15		# learning rate
training_iterations = 400
def h(theta_x):
	# Sigmoid of the value
	return 1.0 / (1.0 + math.exp(-theta_x))

def test_fair_logistic(X, Y, s_id, theta):
	n, d = X.shape

	c_s1_y = 0.0
	c_s2_y = 0.0
	for i in range(n):
		if Y[i] == 0:
			if X[i, s_id] == 0:
				c_s1_y += 1.0
			elif X[i, s_id] == 1:
				c_s2_y += 1.0

	count = 0.0
	first_term_numerator = np.zeros(d, dtype=np.float)
	second_term_numerator = np.zeros(d, dtype=np.float)
	delta_root = 0.0
	delta_true = 0.0
	for i in range(n):
		x_i = X[i,:]
		theta_x = theta.dot(x_i)
		h_x_i = h(theta_x)

		y_bar = 1 if h_x_i >= 0.5 else 0
		if(y_bar == Y[i]):
			count += 1.0
		if(Y[i] == 0):
			if(x_i[s_id] == 0):
				# s1 category
				first_term_numerator += (h_x_i * (1 - h_x_i)) * x_i
				delta_root -= h_x_i / c_s1_y
				delta_true -= y_bar / c_s1_y

			elif(x_i[s_id] == 1):
				# s2 category
				second_term_numerator += (h_x_i * (1 - h_x_i)) * x_i
				delta_root += h_x_i / c_s2_y
				delta_true += y_bar / c_s2_y
	first_term = first_term_numerator / c_s1_y
	second_term = second_term_numerator / c_s2_y
	delta_root = abs(delta_root)
	print "Test:", delta_root, "\t", abs(delta_true), "\t", count / float(n)
	return delta_root, count / float(n)

def train_fair_logistic(X, Y, s_id, X_test, Y_test):
	# We get 3 inputs
	# 1 - X = n feature vectors each of d dimension
	# 2 - Y = n labels corresponding to the features
	# 3 - s_id = column id of the sensitive feature
	# We have to train the fair_logistic loss using full gradient descent

	n, d = X.shape
	theta = np.random.randn(d)

	c_s1_y = 0.0
	c_s2_y = 0.0
	for i in range(n):
		if Y[i] == 0:
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
			like_loss = Y[i] * np.log(h_x_i) + (1-Y[i]) * np.log(1 - h_x_i)
		like_loss_grad /= float(n)
		like_loss /= float(n)
		
		# Calculate Fairness constraint gradients (delta)
		first_term_numerator = np.zeros(d, dtype=np.float)
		second_term_numerator = np.zeros(d, dtype=np.float)
		delta_root = 0.0
		delta_true = 0.0
		for i in range(n):
			x_i = X[i,:]
			theta_x = theta.dot(x_i)
			h_x_i = h(theta_x)

			y_bar = 1 if h_x_i >= 0.5 else 0

			if(Y[i] == 0):
				if(x_i[s_id] == 0):
					# s1 category
					first_term_numerator += (h_x_i * (1 - h_x_i)) * x_i
					delta_root -= h_x_i / c_s1_y
					delta_true -= y_bar / c_s1_y
				elif(x_i[s_id] == 1):
					# s2 category
					second_term_numerator += (h_x_i * (1 - h_x_i)) * x_i
					delta_root += h_x_i / c_s2_y
					delta_true += y_bar / c_s2_y
		first_term = first_term_numerator / c_s1_y
		second_term = second_term_numerator / c_s2_y
		delta_root = abs(delta_root)

		delta_grad = 2.0 * delta_root * (first_term - second_term)

		# Total gradients
		full_grad = -1.0 * like_loss_grad - alpha * delta_grad

		# Updating the parameters
		theta = theta - lr * full_grad

		# Compute accuracy
		correct = 0.0
		for i in range(n):
			x_i = X[i,:]
			theta_x = theta.dot(x_i)
			h_x_i = h(theta_x)
			y_bar = 1 if h_x_i >= 0.5 else 0
			
			if y_bar == Y[i]:
				correct += 1.0
		# print j, "\t", delta_root ** 2,"\t", delta_true, "\t", correct / n
		if (j+1)%5 == 0:
			# print (j+1), "\t", delta_root ** 2, "\t", correct / n
			print (j+1), "\t", delta_root, "\t", -1.0 * like_loss, "\t", correct / n
			# print (j+1), "\t", -1.0 * like_loss, "\t", correct / n
			# test_fair_logistic(X_test, Y_test, s_id, theta)
	# Retrun the learned weights
	print alpha
	return theta

X, y, x_control = data_loader.load_compas_data()

print X.shape
# print X[0:4]
print y.shape
y[y==-1] = 0

x_control = np.array(x_control['race']).reshape((len(x_control['race']), 1))
print np.sum(y)
print np.sum(x_control)
print x_control.shape
X_full = np.append(X, x_control, axis=1)

indices = np.random.permutation(X_full.shape[0])
split_id = int(indices.shape[0] * 0.9)

X_train = X_full[indices[:split_id],:]
y_train = y[indices[:split_id]]
X_test = X_full[indices[split_id:],:]
y_test = y[indices[split_id:]]

theta = train_fair_logistic(X_train, y_train, X_train.shape[1]-1, X_test, y_test)
delta, accuracy = test_fair_logistic(X_test, y_test, X_test.shape[1]-1, theta)
print theta