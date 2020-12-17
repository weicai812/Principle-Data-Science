import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train = pd.read_csv("Princlple DS A201\Titanic.csv") #the path in the computer
train.head()


#view missing value
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#Countplot of people who survived based on their sex.
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


#Fill in missing value
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

#drop cabin
train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)


#check the data after removing the missing value
train.head()

#Converting Categorical Features

sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

train = pd.concat([train,sex,embark],axis=1)


#Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), train['Survived'], test_size=0.30, random_state=101)


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

modelNB = GaussianNB()
# fit the model with the training data
modelNB.fit(X_train,y_train)


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


modelNB = GaussianNB()
# fit the model with the training data
modelNB.fit(X_train,y_train)


y_test  = np.array(list(y_test))

predictions  = np.array(predictions)
df= pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': predictions.flatten()})
df
