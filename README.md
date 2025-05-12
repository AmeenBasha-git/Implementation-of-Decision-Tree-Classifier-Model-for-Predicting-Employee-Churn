# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Ameen Basha A
RegisterNumber: 212224220008
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
print("Null values:\n",data.isnull().sum())
print("Class distribution:\n",data["left"].value_counts())

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])

data.head()

x=data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours",
          "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]

print(x.head())
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

from sklearn import metrics
y_pred =  dt.predict(x_test)

accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

sp=dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
print("Sample_predicts:",sp)
```
## Output:
![image](https://github.com/user-attachments/assets/704756f6-4da0-40f7-b5ca-a4ee05696ed8)

![image](https://github.com/user-attachments/assets/917f032e-a534-4f3e-8106-b9b4f83b7c85)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
