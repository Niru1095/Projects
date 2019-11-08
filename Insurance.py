#Predicting if person will buy Insurance or not
#Simple Logistic Regression
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
df=pd.read_csv('C:\\Users\\admin\\Desktop\\IIHT KHARGHAR\\insurance_data.csv')
df

plt.scatter(df.age,df.bought_insurance)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,test_size=0.1)
X_test
X_train
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

y_cap=model.predict(X_test)
y_cap
model.score(X_test,y_test)
model.predict_proba(X_test)
model.predict_proba(X_test)
