import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

from xgboost import XGBClassifier
clf = XGBClassifier()
clf.fit(X,Y)

y_pred = clf.predict(X)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,Y))

pickle.dump(clf, open('model.pkl','wb'))
