from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
print("data and label size: ",X.shape,y.shape)  # 3类，4个特征

clf = LogisticRegression(penalty='l2',C=0.8,class_weight='balanced',
	                     random_state=1, solver='lbfgs', max_iter=500, 
	                     multi_class='multinomial', n_jobs=1)

clf.fit(X, y)
#print(clf.decision_function(X))
print(clf.score(X,y))
print(clf.predict(X[0:5, :]))       # 每个样本的预测类别
print(clf.predict_proba(X[0:5, :])) # 每个样本每一类的预测概率
print(clf.get_params(deep=True))    # 超参数
print(clf.coef_)       # 参数，shape=(4,3)
print(clf.intercept_)  # 偏置，shape=(3,)
