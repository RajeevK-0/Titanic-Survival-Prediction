
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

titanic_data = pd.read_csv("Titanic.csv")

titanic_data.head()

data = titanic_data.drop(columns=["PassengerId",'Name','Ticket','Cabin'])

data

data.info()

t = data["Age"].copy()

t.isna().sum()



mean = data["Age"].mean()

std = data["Age"].std()

r = np.random.randint(mean-std , mean+std , size=177)

data["Age"].isna().sum()

data.info()

data["Embarked"].isna()

f = data["Embarked"]

f[f.isna()] = "S"

data["Embarked"].isnull().sum()

data.info()

data.head()

data["Sex"] = data["Sex"].map({"male":0 , "female":1})

data["Embarked"]=data["Embarked"].map({'S' : 0 , 'C':1 , 'Q':2})

data.info()

logReg = LogisticRegression()
sv = SVC()
dt = DecisionTreeClassifier()
rd = RandomForestClassifier()
nn = KNeighborsClassifier()

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x = data.drop(columns="Survived")
y = data["Survived"]

xtrain,xtest ,ytrain,ytest = train_test_split(x,y)

scale = StandardScaler()

xtrainNew = scale.fit_transform(xtrain)

xtestNew = scale.transform(xtest)

logReg.fit(xtrainNew , ytrain)

sv.fit(xtrainNew , ytrain)
dt.fit(xtrainNew , ytrain)
rd.fit(xtrainNew , ytrain)
nn.fit(xtrainNew , ytrain)

p1 = logReg.predict(xtestNew)
p2 = sv.predict(xtestNew)
p3 = dt.predict(xtestNew)
p4 = rd.predict(xtestNew)
p5 = nn.predict(xtestNew)

accuracy_score(p1 , ytest)

accuracy_score(p2 , ytest)

accuracy_score(p3 , ytest)

accuracy_score(p4 , ytest)

accuracy_score(p5 , ytest)

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(xtrainNew , ytrain)

p6 = nb.predict(xtestNew)

accuracy_score(p6,ytest)

