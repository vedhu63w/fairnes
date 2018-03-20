import numpy as np
import math
import load_compass as data_loader

alpha = 2		# alpha is the fairness multiplicative constant
lr = 0.1		# learning rate
training_iterations = 100
def h(theta_x):
	# Sigmoid of the value
	return 1.0 / (1.0 + math.exp(-theta_x))

def train_fair_logistic(X, Y, s_id):
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
		if Y[i] == 1:
			if X[i, s_id] == 0:
				c_s1_y += 1.0
			elif X[i, s_id] == 1:
				c_s2_y += 1.0

	for j in range(training_iterations):
		# Calculate Likelihood loss gradients
		like_loss_grad = np.zeros(d, dtype=np.float)
		for i in range(n):
			x_i = X[i,:]
			theta_x = theta.dot(x_i)
			h_x_i = h(theta_x)
			like_loss_grad_i = x_i * (Y[i] - h_x_i)
			like_loss_grad += like_loss_grad_i
		like_loss_grad /= float(n)
		
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

			if(Y[i] == 1):
				if(x_i[s_id] == 0):
					# s1 category
					first_term_numerator += (h_x_i * (1 - h_x_i)) * x_i
					delta_root -= h_x_i / c_s1_y
					delta_true -= y_bar / c_s1_y
				elif(x_i[s_id] == 0):
					# s2 category
					second_term_numerator += (h_x_i * (1 - h_x_i)) * x_i
					delta_root += h_x_i / c_s2_y
					delta_true += y_bar / c_s2_y
		first_term = first_term_numerator / c_s1_y
		second_term = second_term_numerator / c_s2_y
		delta_root = abs(delta_root)

		delta_grad = 2.0 * delta_root * (first_term - second_term)

		# Total gradients
		full_grad = -1.0 * like_loss_grad + alpha * delta_grad

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
		print j, "\t", delta_root ** 2, "\t", correct / n
		print 
	# Retrun the learned weights
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

theta = train_fair_logistic(X_full, y, X_full.shape[1]-1)
print theta