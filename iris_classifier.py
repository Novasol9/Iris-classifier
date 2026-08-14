 #""" 'created 13.082026'"""



import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
iris=load_iris()
desc=iris.DESCR

df=pd.DataFrame(iris.data, columns=iris.feature_names)
df['species']=iris.target

corr=df.corr()


import matplotlib.pyplot as plt
import seaborn as sns
plt.title('Feature Correlation Heatmap - Iris Dataset')

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')

df['species_name']=df['species'].map({0:'setosa',1:'versicolor',2:'virginia'})
sns.pairplot(df,hue='species_name')

plt.show()

from sklearn.model_selection import train_test_split 




X= df.drop(['species_name','species'] ,axis=1)
y=df['species']


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2 )








