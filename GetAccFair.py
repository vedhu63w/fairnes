from sklearn.linear_model		import LogisticRegression 


def get_acc_fair(x_train_y_train, x_test_y_test, sen_attr):
	x_train, y_train = x_train_y_train
	x_test, y_test = x_test_y_test

	LR = LogisticRegression()
	LR.fit(x_train, y_train)
	y_pred = LR.predict(x_test)
	
	acc = sum(1 for i in range(len(x_test)) if y_test[i]==y_pred[i])/ float(len(x_test))
	parity_fair = abs(sum((1 for i in range(len(y_pred)) if (sen_attr[i] and y_pred[i]))) / \
							float(sum((1 for i in range(len(y_pred)) if sen_attr[i])))
						- sum((1 for i in range(len(y_pred)) if (not sen_attr[i] and y_pred[i]))) \
						/ float(sum((1 for i in range(len(y_pred)) if not sen_attr[i]))) )
	return acc, parity_fair
