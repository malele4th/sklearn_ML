# 1. 二分类
## 1.1二元逻辑回归的模型
 - 逻辑回归（Logistics Regression），是分类模型，不是回归模型，主要用于二分类，也可以完成多分类。
 - LR模型，训练速度快，足够解决普通的分类问题。
 - 逻辑函数又称为sigmoid函数

二元LR模型一般形式为：
$$h_{\theta}(x)=g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}$$
其中 $z=\theta^Tx$，$g(z)=\frac{1}{1+e^{-z}}$，$g^{'}(z)=g(z)(1-g(z))$

当 $\theta^Tx>0 时y=1$ ，当$\theta^Tx<0 时 y=0$
## 1.2 二元逻辑回归的损失函数（目标函数）
直接使用平方损失函数，会导致损失函数非凸，优化过程容易陷入局部最优点。
用==似然函数==推导损失函数。

二分类，样本输出只有0或1，则有
$P(y=1|x,\theta)=h_{\theta}(x)$

$P(y=0|x,\theta)=1-h_{\theta}(x)$

合并成一个式子，得到y的概率分布表达式为：
$$P(y|x, \theta)={h_{\theta}(x)}^y{(1-h_{\theta}(x))}^{1-y}$$

似然函数的表达式为（m个样本）：
$$L(\theta)=\prod_{i=1}^m(h_{\theta}(x^{(i)}))^{y^{(i)}}(1-h_{\theta}(x^{(i)}))^{1-y^{(i)}}$$

==对数似然函数取反，即为损失函数==
**<font color=red>最大化似然函数，最小化损失函数<font>**

$$J(\theta)=-ylog(h_{\theta}(x))-(1-y)log(1-h_{\theta}(x)))$$

==称为交叉熵损失函数==

- 当y=0时，$J(\theta)=-log(1-h_{\theta}(x)))$，$h_{\theta}(x)$趋近于0，则损失很小，趋近于1，则损失很大，即惩罚很大
- 当y=1时，$J(\theta)=-log(h_{\theta}(x)))$，$h_{\theta}(x)$趋近于1，则损失很小，趋近于0，则损失很大，即惩罚很大

**m个样本的损失函数：**
$$J(\theta)=-\sum_{i=1}^m(y^{(i)}log(h_{\theta}(x^{(i)}))+(1-y^{(i)})log(1-h_{\theta}(x^{(i)})))$$

## 1.3 二元逻辑回归的损失函数的优化方法
最小化损失函数，常用梯度下降法

对$\theta$向量进行链式法则求导

$$\begin{aligned}\Delta \theta &= \frac{\partial J(\theta)}{\partial \theta}=\frac{-y}{h_{\theta}(x)}h_{\theta}(x)x^T+\frac{1-y}{1-h_{\theta}(x)}(1-h_{\theta}(x))h_{\theta}(x)x^T \\& =(h_{\theta}(x)-y)x^T \end{aligned}
$$

$$\frac{\partial J(\theta)}{\partial \theta_j}=\sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) ) x_{j}^{(i)}$$

参数更新：
$$\theta=\theta-\alpha\Delta\theta^T$$

假设样本数为m，特征维度为n，则 $x$为$n\times m$矩阵，每个列向量是一个样本，$\theta$为$n\times 1$的列向量，y和$h_{\theta}(x)$为$1 \times m$的向量



## 1.4 二元逻辑回归的正则化
常见的有L1正则化、L2正则化

$$J(\theta)=-ylog(h_{\theta}(x))-(1-y)log(1-h_{\theta}(x)))+\alpha||\theta||_1$$

$$J(\theta)=-ylog(h_{\theta}(x))-(1-y)log(1-h_{\theta}(x)))+\frac{1}{2}\alpha||\theta||^2_2$$

L1正则化，损失函数的优化方法常用的有：坐标轴下降法、最小角回归法
L2正则化，损失函数的优化方法：梯度下降法

# 2. 多分类
## 2.1 多元逻辑回归的概率分布
对于多元逻辑回归的处理方法
- 认为某类为1，其余类为0，称为 one-vs-rest
- 每次选择两类样本做二分类，称为one-vs-one

**对于二分类的推导，y的输出只有0或1**
$$P(y=1 | x, \theta)=h_{\theta}(x)=\frac{1}{1+e^{-\theta^Tx}}=\frac{e^{\theta^Tx}}{1+e^{\theta^Tx}}$$

$$P(y=0 | x, \theta)=1-h_{\theta}(x)=\frac{1}{1+e^{\theta^Tx}}$$

$$\log \frac{P(y=1 | x, \theta)}{P(y=0 | x, \theta)}=\theta^Tx$$

**对于多分类，样本输出y的取值假设为 1，2，...，K**

**看作多个二分类，以第 K 类为基准，则有**

$$\log \frac{P(y=1 | x, \theta)}{P(y=K | x, \theta)}=\theta^T_1x$$

$$\log \frac{P(y=i | x, \theta)}{P(y=K | x, \theta)}=\theta^T_ix$$

$$\log \frac{P(y=K-1 | x, \theta)}{P(y=K | x, \theta)}=\theta^T_{K-1}x$$

由 $\sum_{i=1}^{K} P(y=i | x, \theta)=1$ 可得：

$$(1+\sum_{i=1}^{K} e^{\theta^T_ix})P(y=K|x,\theta)=1$$

得到多元逻辑回归的概率分布为：
$$P(y=K | x, \theta)=\frac{1}{1+\sum_{i=1}^{K-1} e^{\theta_i^T x}}$$

$$P(y=i | x, \theta)=\frac{ e^{\theta_i^T x}}{1+\sum_{i=1}^{K-1} e^{\theta_i^T x}}$$

# 3. sklearn.linear_model.LogisticRegression
## 3.1 sklearn LR模型参数详解
```python
class LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, C=1.0, 
fit_intercept=True, intercept_scaling=1, class_weight=None,
random_state=None, solver=’warn’, max_iter=100, multi_class=’warn’,
verbose=0, warm_start=False, n_jobs=None)
```

参数说明：
==1、penalty==
- 正则项，可选L1或L2。
- 如果使用L2正则化依然过拟合，或者特征非常多，希望做特征选择，可以考虑L1正则化。
- 正则化的选择会影响损失函数优化算法的选择，L2正则化，{newton-cg，lbfgs，liblinear，sag}都可以选择，L1正则化只能选择 liblinear优化算法。因为L1正则化的损失函数不是连续可导的，而{newton-cg，lbfgs，sag}都需要损失函数一阶或二阶连续导数

==2、dual==
- 用来指明是否将原问题转换成对偶问题
- 对偶问题类似于相反问题，比如求解最大值的线性规划，转换成求解最小值的线性规划
- 当样本数大于特征数，选择False，并且只能在liblinear+l2 penalty情况下使用

==3、tol==
- 残差收敛条件，收敛的时候两步之差<tol就停止。
 
 ==4、C==
- 正则化系数，必须是正数，值越小，正则化强度越大，防止过拟合的程度越大，$\alpha=1/C$

==5、fit_intercept==
- 指定是否将常量（bias or intercept）加入到 decision function

==6、intercept_scaling==
- 仅在优化算法 liblinear 情况下使用
- x变成[x, self.intercept_scaling]，就是w0或者b

==7、class_weight==
- 调节正负样本比例，默认为None，可以选择 “balanced”，可以传入字典 {class_label:weight}
- balanced，自动调整与输入数据中的类频率成反比的权重
- 如果制定了sample_weight，class_weight将与sample_weight相乘

==8、random_state==
- 随机种子的设置，使用相同的训练集和测试集，运行多少次，结果都一样

==9、solver==
- 损失函数的优化算法，{liblinear，lbfgs，newton-cg，sag，saga}
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190507135244551.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21hbGVsZTR0aA==,size_16,color_FFFFFF,t_70)
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

==12、verbose==
- "冗余的"，会输出一些模型运算过程，默认0，不输出
- 对于liblinear，lbfgs，设置任意正数，输出详细过程

==13、warm_start==
- 是否使用上次的模型结果作为初始化

==14、n_jobs==
- 并行化时使用的CPU核的数量，-1表示使用所有的CPU核

## 3.2 sklearn LR模型的对象和方法
模型对象
- coef_:返回各特征的系数,绝对值大小可以理解成特征重要性
- intercept_:返回模型的截距，即偏置

模型方法
- decision_function(X):返回决策函数值（比如svm中的决策距离）
- predict_proba(X):返回每个类别的概率值（有几类就返回几列值）
- predict_log_proba(X):返回概率值的log值（即将概率取对数）
- predict(X)：返回预测结果值（0/1）
- score(X, y=None):返回正确率
- get_params(deep=True):返回估计器的参数

## 3.3 sklearn LR模型使用例子
```python
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
```
输出结果为：

```shell
data and label size:  (150, 4) (150,)                                                                                                                                   
0.9733333333333334                                                                                                                                                      
[0 0 0 0 0]                                                                                                                                                             
[[9.78571369e-01 2.14285809e-02 4.97036800e-08]                                                                                                                         
 [9.67528768e-01 3.24711382e-02 9.34656464e-08]                                                                                                                         
 [9.82653272e-01 1.73466871e-02 4.06294555e-08]                                                                                                                         
 [9.72519756e-01 2.74801246e-02 1.19497174e-07]                                                                                                                         
 [9.82584586e-01 1.74153725e-02 4.17372134e-08]]                                                                                                                        
{'C': 0.8, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 
'max_iter': 500, 'multi_class': 'multinomial', 'n_jobs': 1, 'penalt
y': 'l2', 'random_state': 1, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}                                                                       
[[-0.41970458  0.88110727 -2.36861993 -1.01249304]                                                                                                                      
 [ 0.48079601 -0.3330422  -0.17330735 -0.84848953]                                                                                                                      
 [-0.06109144 -0.54806507  2.54192728  1.86098257]]                                                                                                                     
[  9.44748223   2.1768459  -11.62432814]                                                                                                                                
```

**参考博客**
[1. 逻辑回归原理小结，刘建平](https://www.cnblogs.com/pinard/p/6029432.html)
[2. 多元逻辑回归公式推导](https://blog.csdn.net/huanghongyi123456/article/details/86504709)
[3. 逻辑回归(logistics regression)](https://blog.csdn.net/weixin_39445556/article/details/83930186)
[4. sklearn LR模型文档](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
[5. sklearn LR模型源码](https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/linear_model/logistic.py)
[6. Sklearn参数详解—LR模型](https://cloud.tencent.com/developer/article/1146080)
