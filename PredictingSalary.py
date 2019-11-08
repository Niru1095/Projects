import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

ds=pd.read_csv('C:\\Users\\admin\\Desktop\\IIHT KHARGHAR\\ML\\Projects\\hiring.csv')
ds.head()

ds['experience']=ds.experience.fillna('zero')
ds.head()

pip install word2number
from word2number import w2n
ds['experience']=ds.experience.apply(w2n.word_to_num)
ds.head(8)
ds.info()

import math
median_test_score=math.floor(ds["test_score(out of 10)"].median())
median_test_score

ds["test_score(out of 10)"]=ds["test_score(out of 10)"].fillna(median_test_score)
x=ds.drop('salary($)',axis=1)
x.shape
y=ds['salary($)']
y.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)
ypred=reg.predict(X_test)
reg.score(X_test,y_test)

reg.coef_
reg.intercept_
reg.fit(ds[['experience','test_score(out of 10)','interview_score(out of 10)']],ds['salary($)'])
reg.predict([[0,7,8]])
reg.predict([[5,6,7]])
reg.predict([[2,9,6]])

plt.scatter(ds.experience,ds['salary($)'])
#plt.plot(X_train,2812.95*X_train+1845.70*X_train+2205.24*X_train+17737.26,'r')
