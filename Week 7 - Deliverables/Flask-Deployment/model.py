# Importing the libraries
import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('Bank.csv')



X = dataset['age','job','marital','education','balance','housing','loan','contact','month','default']


y = dataset.iloc['y']

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 2200, 5]]))