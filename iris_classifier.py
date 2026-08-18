 #""" 'created 13.082026'"""



import numpy as np
import pandas as pd

from sklearn.datasets import load_iris

iris=load_iris()
desc=iris.DESCR
df=pd.DataFrame(iris.data, columns=iris.feature_names)
df['species']=iris.target

corr=df.corr()


# -----------------------------------features pairplot and heatmap--------------------------

import matplotlib.pyplot as plt
import seaborn as sns

df['species_name']=df['species'].map({0:'setosa',1:'versicolor',2:'virginia'})
sns.pairplot(df,hue='species_name')
plt.show()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap - Iris Dataset')
plt.show()


from sklearn.model_selection import train_test_split 

X= df.drop(['species_name','species'] ,axis=1)
y=df['species']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2 )

# --------------------------------------Logistic regression-----------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model=LogisticRegression(max_iter=200)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test, y_pred)
classfn_report=classification_report(y_test, y_pred)


# ---------------------------------Cross-validation-score-------------------------------------

from sklearn.model_selection import cross_val_score

scores=cross_val_score(model, X, y,cv=5)
std_dev=scores.std()
mean_acc=scores.mean()



