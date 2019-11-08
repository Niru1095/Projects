import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ads=pd.read_csv('C:\\Users\\admin\\Desktop\\IIHT KHARGHAR\\ML\\Projects\\advertising.csv')

ads.head()
ads.shape
ads.info()
ads.describe()
ads.isnull().sum()

fig,axs=plt.subplots(3,figsize=(5,5))
plt1=sns.boxplot(ads['TV'],ax=axs[0])
plt2=sns.boxplot(ads['Radio'],ax=axs[1])
plt2=sns.boxplot(ads['Newspaper'],ax=axs[2])
plt.tight_layout()

sns.boxplot(ads['Sales'])

sns.pairplot(ads,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',height=4,aspect=1,kind='scatter')

sns.heatmap(ads.corr(), cmap="YlGnBu", annot = True)
plt.show()
ads.info()
x=ads.drop(['Radio','Newspaper','Sales'],axis=1)
x.shape
y=ads['Sales']
y.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

X_train.head()
y_train.head()
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
model.score(X_test,y_test)
model.coef_
model.intercept_

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

np.sqrt(mean_squared_error(y_test,y_pred))
r_squared=r2_score(y_test,y_pred)
r_squared

plt.scatter(X_test,y_test)
plt.plot(X_train,0.054*X_train+7.143,'r')
#OTHER METHOD_Linear Regression
import statsmodels.api as sm
X_train_sm=sm.add_constant(X_train)
lr=sm.OLS(y_train,X_train_sm).fit()
lr.params
lr.summary()

plt.scatter(X_train,y_train)
plt.plot(X_train,0.054*X_train+7.143,'r')

y_train_pred=lr.predict(X_train_sm)
res=(y_train-y_train_pred)

fig=plt.figure()
sns.distplot(res,bins=15)
fig.suptitle('Error Terms')
plt.xlabel('y_train-y-train_pred')

plt.scatter(X_train,res)

X_test_sm=sm.add_constant(X_test)
y_pred=lr.predict(X_test_sm)

