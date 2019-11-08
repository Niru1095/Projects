import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("C:\\Users\\admin\\Desktop\\IIHT KHARGHAR\\ML\\Projects\\canada_per_capita_income.csv")
df.head()
df.isnull().sum()

df.info()
plt.xlabel("Year")
plt.ylabel("per Capita Income (US$)")
plt.scatter(df["year"],df["per capita income (US$)"])

x=df.drop('per capita income (US$)',axis='columns')
x.shape
y=df["per capita income (US$)"]
y.shape
new_df = df.drop('per capita income (US$)',axis='columns')
new_df.shape
new_income=df["per capita income (US$)"]
new_income.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=15)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
X_train.head()
y_train.head()
model.fit(X_train,y_train)
model.score(X_test,y_test)
model.fit(new_df,new_income)
model.predict([[2016]])

model.coef_
model.intercept_
plt.scatter(X_test,y_test)
plt.plot(X_train,828.46*X_train-1632210.75,'r')
