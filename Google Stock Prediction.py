import math
import pickle
import quandl
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_Percent'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100.0
df['Percent_Change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_Percent', 'Percent_Change', 'Adj. Volume']]

forcast_col = 'Adj. Close' #Column used for Prediction/Forcast
df.fillna(-99999, inplace = True) #Filling Enpty Cells with a random value

forcast_out = int(math.ceil(0.01 * len(df))) #n(35) Entries used for Next Prediction of Forcast Col

df['Label'] = df[forcast_col].shift(- forcast_out) #Value of Forcast Column after n(35) days into the Future
df.dropna(inplace = True)

X = np.array(df.drop(['Label'],1))
X = preprocessing.scale(X)
X_lately = X[-forcast_out:] #To have X values for next n(35) days.

df.dropna(inplace = True)
y = np.array(df['Label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size = 0.2)

clf = LinearRegression(n_jobs = 10) #Running 10 threads simultaneously to reduce training time. Default n_jobs = 1, if n_jobs = -1 then maximum number of threads
clf.fit(X_train,y_train)

with open('GoogleStockPredictionLinearRegressionClassifier.pickle','wb') as f: #Using Pickle to save the Trained Classifier to save the time on retraining it every time code runs
    pickle.dump(clf, f)

pickle_in = open('GoogleStockPredictionLinearRegressionClassifier.pickle','rb') #Opening the Trained Classifier using Pickle
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test) #Simple Linear Regression Accuracy

'''
clf2 = svm.SVR()
clf2.fit(X_train,y_train)
accuracy2 = clf2.score(X_test, y_test) #SVM SVR(Support Vector Regression) Accuracy

clf3 = svm.SVR(kernel = 'poly')
clf3.fit(X_train,y_train)
accuracy3 = clf3.score(X_test, y_test) #SVM SVR(Support Vector Regression, Kernel is polynomial) Accuracy
'''

forcast_set = clf.predict(X_lately) #Predicted values of next n(35) days

style.use('ggplot')

df['Forcast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forcast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i] #Substitutes the forcast value(i) for all the attributes in the Future and replaces all other attribute values with NaN 

df['Adj. Close'].plot()
df['Forcast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')

print(accuracy)
print(df.head())
print(df.tail())

plt.show()
