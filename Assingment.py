import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing csv
train=pd.read_csv('C:\\Users\\admin\\Desktop\\IIHT KHARGHAR\\PYTHON\\loan_data_train.csv')
train
test=pd.read_csv('C:\\Users\\admin\\Desktop\\IIHT KHARGHAR\\PYTHON\\loan_data_test.csv')
test

#fidnding names of columns because train and test  NO. of columns don't match
names_train=(train.columns)
names_test=(test.columns)

#finding which particular column is not there in test dataset
list(filter(lambda x:x not in names_test,names_train))

test['Interest.Rate']='NA'

train['Data']='train'
test['Data']='test'

ds=pd.concat([train,test])

ds.info
ds.describe
ds.dtypes

#Converting Object to float Dtypes
ds['Amount.Funded.By.Investors']=pd.to_numeric(ds['Amount.Funded.By.Investors'].str.replace('.',''),downcast='float')
ds.dtypes

ds['Amount.Requested']=pd.to_numeric(ds['Amount.Requested'].str.replace('.',''),downcast='float')
ds.dtypes

ds['Debt.To.Income.Ratio']=pd.to_numeric(ds['Debt.To.Income.Ratio'].str.replace('%',''),downcast='float')
ds.dtypes

ds['Employment.Length'].value_counts()
for i in range(len(ds['Employment.Length'])):
    ds['Employment.Length']=ds['Employment.Length'].str.replace('years','')
    ds['Employment.Length']=ds['Employment.Length'].str.replace('year','')
    ds['Employment.Length']=ds['Employment.Length'].str.replace('10\\+','10')
    ds['Employment.Length']=ds['Employment.Length'].str.replace('.','')
    ds['Employment.Length']=ds['Employment.Length'].str.replace('< 1','0')
   
ds['Employment.Length'].value_counts()
ds['Employment.Length']=pd.to_numeric(ds['Employment.Length'],downcast='integer')    
ds.dtypes

ds[['f1','f2']]=ds['FICO.Range'].str.split('-',expand=True)
ds['f1']=pd.to_numeric(ds['f1'],downcast='integer')
ds['f2']=pd.to_numeric(ds['f2'],downcast='integer')
ds['Fico']=0.5*(ds['f1']+ds['f2'])
ds['Fico']
ds.dtypes
drop_cols=['f1','f2','FICO.Range','ID']
ds.drop(drop_cols,axis=1,inplace=True)
ds.dtypes

for i in range(len(ds['Interest.Rate'])):
    ds['Interest.Rate']=ds['Interest.Rate'].str.replace('NA','')
    ds['Interest.Rate']=ds['Interest.Rate'].str.replace('%','')
    
ds['Interest.Rate']=pd.to_numeric(ds['Interest.Rate'],downcast='float')
ds.dtypes

ds['Open.CREDIT.Lines']=pd.to_numeric(ds['Open.CREDIT.Lines'].str.replace('.',''),downcast='integer')
ds.dtypes

ds['Revolving.CREDIT.Balance']=pd.to_numeric(ds['Revolving.CREDIT.Balance'].str.replace('.',''),downcast='integer')
ds['Loan.Length'].value_counts()
ds['Loan.Length']=ds['Loan.Length'].replace('.',np.nan)
ds['Loan.Length'].isnull().sum()

from sklearn.impute import SimpleImputer# new codes
si=SimpleImputer(missing_values=np.nan,strategy='most_frequent')#'mean', “median”, “most_frequent”,“constant”
si=si.fit(ds[['Loan.Length']])# [[]] because simple imputer takes data frame as input
ds['Loan.Length']=si.transform(ds[['Loan.Length']])

ds['Loan.Length'].value_counts()

ds['Home.Ownership'].value_counts()
ds['Home.Ownership'].isnull().sum()

si=SimpleImputer(missing_values=np.nan,strategy='most_frequent')#'mean', “median”, “most_frequent”,“constant”
si=si.fit(ds[['Home.Ownership']])# [[]] because simple imputer takes data frame as input
ds['Home.Ownership']=si.transform(ds[['Home.Ownership']])


ds=pd.get_dummies(ds,columns=['Loan.Length','Home.Ownership'],drop_first=True)
ds.info()

ds['State'].value_counts()
ds['State']=ds['State'].replace('.','np.nan')
ds['State'].isnull().sum()

from sklearn.impute import SimpleImputer
si=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
si=si.fit(ds[['State']])
ds['State']=si.transform(ds[['State']])

ds['Loan.Purpose'].value_counts()
ds['Loan.Purpose'].isnull().sum()

si=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
si=si.fit(ds[['Loan.Purpose']])
ds['Loan.Purpose']=si.transform(ds[['Loan.Purpose']])

ds=pd.get_dummies(ds,columns=['Loan.Purpose','State'],drop_first=True)
ds.dtypes

ds.isnull().sum()

null_variables=['Amount.Funded.By.Investors','Amount.Requested','Debt.To.Income.Ratio','Employment.Length',
                'Inquiries.in.the.Last.6.Months','Monthly.Income','Open.CREDIT.Lines',
                'Revolving.CREDIT.Balance']

from sklearn.impute import SimpleImputer# new codes
si=SimpleImputer(missing_values=np.nan,strategy='mean')
si=si.fit(ds[null_variables])
ds[null_variables]=si.transform(ds[null_variables])

ds.isnull().sum()

ds_train=ds[ds['Data']=='train']
ds_test=ds[ds['Data']=='test']

ds_train.drop('Data',axis=1,inplace=True)
ds_test.drop('Data',axis=1,inplace=True)

ds_train.dtypes

X_train=ds_train.loc[:,ds_train.columns !='Interest.Rate']
Y_train=ds_train['Interest.Rate']

X_test=ds_train.loc[:,ds_test.columns !='Interest.Rate']
Y_test=ds_test['Interest.Rate']


#######################################
x=pd.concat([X_train,X_test])
y=pd.concat([Y_train,Y_test])
y.shape
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2)
