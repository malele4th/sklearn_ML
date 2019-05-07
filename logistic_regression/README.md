## sklearn LR模型参数详解
```python
class LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, C=1.0, 
fit_intercept=True, intercept_scaling=1, class_weight=None,
random_state=None, solver=’warn’, max_iter=100, multi_class=’warn’,
verbose=0, warm_start=False, n_jobs=None)
```

==1、penalty==
- 正则项，可选L1或L2。
- 如果使用L2正则化依然过拟合，或者特征非常多，希望做特征选择，可以考虑L1正则化。
- 正则化的选择会影响损失函数优化算法的选择，L2正则化，{newton-cg，lbfgs，liblinear，sag}都可以选择，L1正则化只能选择 liblinear优化算法。因为L1正则化的损失函数不是连续可导的，而{newton-cg，lbfgs，sag}都需要损失函数一阶或二阶连续导数
 
 ==4、C==
- 正则化系数，必须是正数，值越小，正则化强度越大

==7、class_weight==
- 调节正负样本比例，默认为None，可以选择 “balanced”，可以传入字典 {class_label:weight}
- balanced，自动调整与输入数据中的类频率成反比的权重
- 如果制定了sample_weight，class_weight将与sample_weight相乘


==9、solver==
- 损失函数的优化算法，{liblinear，lbfgs，newton-cg，sag，saga}
- newton-cg, lbfgs和sag 只能用于L2正则化。
- liblinear对L1正则化和L2正则化都适用。
- sag，数据量较少时不宜选用，数据量很大时，为了速度，sag是第一选择。

==10、max_iter==
- 优化算法收敛的最大迭代次数，适用于{newton-cg，lbfgs，sag}

==11、multi_class==
- 可选 ovr、auto和multinomial，二分类问题效果一样，主要用于多分类问题
- 如果二分类，或者liblinear优化算法，自动选择ovr，其他选择 multinational
- ovr：把分类问题简化成是/非两类问题 one-vs-rest
- multinomial：每次选出两个类别进行判断，再从剩下的类别中选出两个进行判断 one-vs-one



## sklearn LR模型的对象和方法
模型对象
- coef_:返回各特征的系数,绝对值大小可以理解成特征重要性
- intercept_:返回模型的截距，即偏置

模型方法
- predict_proba(X):返回每个类别的概率值（有几类就返回几列值）
- predict(X)：返回预测结果值（0/1）
- score(X, y=None):返回正确率                                                      

