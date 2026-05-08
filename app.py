import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

#load dataset
train = pd.read_csv("D:\Data Science\Loan Prediction\Loan Train.csv")
test = pd.read_csv("D:\Data Science\Loan Prediction\Loan Test.csv")

train_original=train.copy()
test_original=test.copy()

#Exploratory data analysis
train.dtypes
train.shape
train['Loan_Status'].value_counts()
train['Loan_Status'].value_counts(normalize=True)
train['Loan_Status'].value_counts().plot.bar(title='Loan_Status')

plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(5,5), title='Gender')

plt.figure(2)
plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(figsize=(6,6),title='Married')

plt.figure(3)
plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(figsize=(6,6),title='Self_Employed')

plt.subplot(223)
train['Credit_History'].value_counts(normalize=True).plot.bar(figsize=(6,6),title='Credit_History')

train['Dependents'].value_counts(normalize=True).plot.bar(title='Dependents')

plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(title='Education')

plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title='Property_Area')

plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16,5))
plt.show()

train.boxplot(column='ApplicantIncome', by = 'Education')
plt.suptitle("ApplicantIncome with Education Qualification")

plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(16,5))
plt.show()

plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,5))
plt.show()

Gender = pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True, figsize=(4,4))             

Married = pd.crosstab(train['Married'],train['Loan_Status'])
Married.div(Married.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))

Dependents = pd.crosstab(train['Dependents'],train['Loan_Status'])
Dependents.div(Dependents.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()

Education = pd.crosstab(train['Education'],train['Loan_Status'])
Education.div(Education.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()

Self_Employed = pd.crosstab(train['Self_Employed'],train['Loan_Status'])
Self_Employed.div(Self_Employed.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()

Credit_History = pd.crosstab(train['Credit_History'],train['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()

Property_Area = pd.crosstab(train['Property_Area'],train['Loan_Status'])
Property_Area.div(Property_Area.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
bins = [0, 2500, 4000, 6000, 81000]
group = ['Low', 'Average', 'High', 'Very high']

train['Income_bin'] = pd.cut(train['ApplicantIncome'], bins, labels=group)

Income_bin = pd.crosstab(train['Income_bin'], train['Loan_Status'])

Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.xlabel('ApplicantIncome')
plt.ylabel('Percentage')

plt.show()

bins=[0,1000,3000,42000]
group=['Low','Average','High']
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)
Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('CoapplicantIncome')
plt.ylabel('Percentage')

train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
bins=[0,2500,4000,6000,81000]
group=['Low','Average','High','Very High']
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)
Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Total_Income')
p=plt.ylabel('Percentage')

bins=[0,1000,3000,42000]
group=['Low','Average','High']
train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('LoanAmount')
p = plt.ylabel('Percentage')

train.isnull().sum()

#missing value imputations
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

train['Loan_Amount_Term'].value_counts()

train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

train.isnull().sum()

#missing value imputations
test['Gender'].fillna(test['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(test['LoanAmount'].median(), inplace=True)

train = train.drop(['Income_bin','Coapplicant_Income_bin','LoanAmount_bin','Total_Income'], axis=1, errors='ignore')

train['Dependents'].replace('3+', 3, inplace=True)

train['Loan_Status'].replace({'N':0, 'Y':1}, inplace=True)

print(train.columns)

train = train.drop(columns=['Loan_ID'], errors='ignore')
test = test.drop(columns=['Loan_ID'], errors='ignore')

X = train.drop('Loan_Status',axis=1)
y = train.Loan_Status

X = pd.get_dummies(X)
train = pd.get_dummies(train)
test = pd.get_dummies(test)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(train.corr(), vmin=-1, vmax=1)  # fixed here
fig.colorbar(cax)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_cv, y_train, y_cv = train_test_split(X,y, test_size=0.3)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(X_train,y_train)

pred_cv = model.predict(X_cv)
accuracy_score(y_cv,pred_cv)

from sklearn import metrics
import matplotlib.pyplot as plt

# Use predicted probabilities (better)
pred_prob = rf.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_prob)
auc = metrics.roc_auc_score(y_test, pred_prob)

plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, label="AUC = " + str(auc))

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend(loc=3)
plt.show()




