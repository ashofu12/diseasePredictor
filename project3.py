import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


df = pd.read_csv("sympton_dataset.csv")

def sortlist(j):
    k = []
    for i in j:
        if i == i:
            k.append(i)
        else:
            break
        
    k.sort()
    l = len(j)-len(k)
    for i in range(l):
        k.append(np.NaN)
    return k

#print(df.head())


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

x = df.drop("Disease", axis=1)

symptoms = df.drop(["Disease"], axis=1).values.tolist()
symptoms = list(np.concatenate(symptoms).flat)
symptoms = list(set(symptoms))
symptoms.remove("nan")
symptoms.sort()
#symptoms = symptoms[:5]

print("For the next 131 symptoms, please enter \"yes\" if you have it, and skip if you haven't or don't know")
input("Press enter to start")


patient = [np.NaN]*17
s = 0

for i in range(len(symptoms)):
    ans = input(symptoms[i].capitalize().replace("_", " ")+": ")
    if ans.lower() == "yes":
        patient[s] = symptoms[i]
        s+=1

print("Please wait...")
x.loc[len(df.index)] = patient



#print(x.head())
#print(patient)

for i in range(x.shape[0]):
    d = x.iloc[i]
    j = sortlist(d)
    x.loc[i] = j

for i in x.columns:
    x[i] = le.fit_transform(x[i].astype(str))
#print(x.head())

test = x.iloc[-1:]
#print(test)
#print(df.shape)
x = x[:-1]
#print(x.shape[0])


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import *

y = df[["Disease"]]

#print(y.shape)
#print(y.shape[0])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, stratify=None)
#print(x_test)

#random forest *most accurate*
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
#print(rf.predict(x_test))
#*print("random forest: ", rf.score(x_test, y_test))
#plot_confusion_matrix(rf, x_test, y_test)
#plt.show()
rf.fit(x, y)
#print(x_test)
#print(test)
z = rf.predict(test)[0]
print("\nYou most likely have "+z)

#decision tree
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
#print("decision tree: ", dt.score(x_test, y_test))

#KNN
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
#print("knn: ", knn.score(x_test, y_test))

#gaussian NB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
#print("gaussion NB: ", gnb.score(x_test, y_test))

#neural network
nn = MLPClassifier()
nn.fit(x_train, y_train)
#print("neural network: ", nn.score(x_test, y_test))

