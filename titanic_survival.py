import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

train=pd.read_csv("C:\\Users\\admin\\Desktop\\IIHT KHARGHAR\\ML\\Projects\\Titanic\\train.csv")
test=pd.read_csv("C:\\Users\\admin\\Desktop\\IIHT KHARGHAR\\ML\\Projects\\Titanic\\test.csv")

train.head()
train.info()
train.shape

test.info()
test.shape

train['SibSp'].value_counts()
train['Pclass'].value_counts()
train['Parch'].value_counts()

train.isnull().sum()
test.isnull().sum()

def barchart(feature):
    survived=train[train['Survived']==1][feature].value_counts()
    dead=train[train['Survived']==0][feature].value_counts()
    df=pd.DataFrame([survived,dead])
    df.index=['Survived','Dead']
    df.plot(kind='bar',stacked=True,figsize=(10,5))
    
barchart('Sex')
barchart('Pclass')
barchart('SibSp')
barchart('Parch')
barchart('Embarked')

train_test_data=[train,test]
    
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    

train['Title'].value_counts()
test['Title'].value_counts()

title_mapping={"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Dr": 3, "Rev": 3, "Col": 3,
               "Major": 3, "Mlle": 3,"Countess": 3,"Ms": 3, "Lady": 3, "Jonkheer": 3,
               "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:
    dataset["Title"]=dataset["Title"].map(title_mapping)
    
train.head()
train.info()
test.head()

barchart("Title")
train.drop("Name",axis=1,inplace=True)
test.drop("Name",axis=1,inplace=True)
test.info()
train.info()

sex_mapping={"male":0,"female":1}
for dataset in train_test_data:
    dataset["Sex"]=dataset["Sex"].map(sex_mapping)
    
train.head()
barchart("Sex")

xtrain.info()
train["Age"].isnull().sum()

from sklearn.impute import SimpleImputer
si=SimpleImputer(missing_values=np.nan,strategy="median")
si=si.fit(train[['Age']])
train['Age']=si.transform(train[['Age']])
train["Age"].isnull().sum()

from sklearn.impute import SimpleImputer
si=SimpleImputer(missing_values=np.nan,strategy="median")
si=si.fit(test[['Age']])
test['Age']=si.transform(test[['Age']])
test['Age'].isnull().sum()

#test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)

facet=sns.FacetGrid(train,hue="Survived",aspect=2)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0,train["Age"].max()))
facet.add_legend()
plt.show()

facet=sns.FacetGrid(train,hue="Survived",aspect=2)
facet.map(sns.kdeplot,"Age",shade=True)
facet.set(xlim=(0,train["Age"].max()))
facet.add_legend()
plt.xlim(0,20)

train.info()
test.info()

for dataset in train_test_data:
    dataset.loc[dataset['Age']<=18,'Age']=0
    dataset.loc[(dataset['Age']>18) & (dataset['Age']<=45),'Age']=1
    dataset.loc[(dataset['Age']>45)& (dataset['Age']<=60),'Age']=2
    dataset.loc[dataset['Age']>60,'Age']=3
    
barchart('Age')

train.info()
train['Embarked'].value_counts()
train['Embarked'].isnull().sum()
test['Embarked'].isnull().sum()

from sklearn.impute import SimpleImputer
si=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
si=si.fit(train[['Embarked']])
train['Embarked']=si.transform(train[['Embarked']])

Embarked_mapping={'S':0,'C':1,'Q':2}
for dataset in train_test_data:
    dataset['Embarked']=dataset['Embarked'].map(Embarked_mapping)

barchart('Embarked')
train.head()
train.info()
test.info()
test['Fare'].isnull().sum()

test['Fare'].fillna(test.groupby("Pclass")["Fare"].transform("median"),inplace=True)


facet=sns.FacetGrid(train,hue="Survived",aspect=2)
facet.map(sns.kdeplot,'Fare',shade=True)
facet.set(xlim=(0,train['Fare'].max()))
facet.add_legend()
plt.show()

train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)

train.drop('Ticket',axis=1,inplace=True)
test.drop('Ticket',axis=1,inplace=True)

train.drop('PassengerId',axis=1,inplace=True)
test.drop('PassengerId',axis=1,inplace=True)

train_data=train.drop("Survived",axis=1)
target=train["Survived"]

train_data.info()


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold=KFold(n_splits=10,shuffle=True,random_state=0)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
score=cross_val_score(model,train_data,target,cv=k_fold,n_jobs=1,scoring='accuracy')
print(score)

round(np.mean(score)*100,2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=13)
score=cross_val_score(model,train_data,target,cv=k_fold,n_jobs=1,scoring='accuracy')
print(score)

round(score.mean()*100,2)

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
score=cross_val_score(model,train_data,target,cv=k_fold,n_jobs=1,scoring='accuracy')
print(score)

round(score.mean()*100,2)

from sklearn.svm import SVC
model=SVC()
score=cross_val_score(model,train_data,target,cv=k_fold,n_jobs=1,scoring='accuracy')
print(score)

round(score.mean()*100,2)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
score=cross_val_score(model,train_data,target,cv=k_fold,n_jobs=1,scoring='accuracy')
print(score)

round(score.mean()*100,2)

model=DecisionTreeClassifier()
model.fit(train_data,target)
test_data=test.drop("PassengerId",axis=1).copy()
prediction=model.predict(test_data)

result=pd.DataFrame({"PassengerId":test["PassengerId"],"Survived":prediction})
result
