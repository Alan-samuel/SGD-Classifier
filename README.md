# Exp.No:7-- SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Predict Iris Species using SGD Classifier.
   
2.Load the Dataset.

3.Preprocess the Data.
 
4.Train the SGD Classifier and make Predictions.

5.Evaluate the Model. 

## Program:
~~~
Developed by: Alan Samuel Vedanayagam
RegisterNumber: 212223040012
~~~
~~~
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
iris=load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgd_clf.fit(X_train,y_train)
y_pred=sgd_clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
~~~



## Output:

![image](https://github.com/user-attachments/assets/f0285900-4ac0-40aa-a6f0-1eb30764b896)
![image](https://github.com/user-attachments/assets/c6a4f336-a053-4741-8cd2-03210fa25215)
![image](https://github.com/user-attachments/assets/853054a2-ec6a-4f6a-a15b-76362f7cd4b5)
![image](https://github.com/user-attachments/assets/9027b043-d115-46e7-96c8-da9ab53f5fb8)




## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
