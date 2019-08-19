import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import visuals as vs
from sklearn.model_selection import ShuffleSplit,GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt








data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV',axis = 1)
features['ff'] = features['RM']*features['LSTAT']

plt.plot(features['RM'],prices,'r--',features['LSTAT'],prices,'bs',features['ff'],prices,'g^')

plt.show()
print("price dec",prices.head())
print("features dec",features.head())

def performance_metric(y_true,y_predict):

    score = r2_score(y_true,y_predict)

    return score

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.3,shuffle=True)

##vs.ModelLearning(features, prices)
##vs.ModelComplexity(X_train, y_train)

def fit_model(X,Y):

    cv_sets = ShuffleSplit(n_splits=10,test_size=0.2)

    regressor = DecisionTreeRegressor()

    parms={'max_depth':[1,2,3,4,5,6,7,8,9,10]}

    scoring_fun = make_scorer(performance_metric)

    grid = GridSearchCV(regressor,parms,scoring=scoring_fun,cv=cv_sets)

    grid.fit(X,Y)

    return grid.best_estimator_

reg = fit_model(X_train,y_train)


print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()))

predict_test = reg.predict(X_test)

acc = performance_metric(y_test,predict_test)

print("acc = {:.3f}".format(acc))

def fit_model1(X,Y):

    cv_sets = ShuffleSplit(n_splits=10,test_size=0.2)

    regressor = LinearRegression()

    parms={'normalize':[True,False],'fit_intercept':[True,False]}

    scoring_fun = make_scorer(performance_metric)

    grid = GridSearchCV(regressor,parms,scoring=scoring_fun,cv=cv_sets)

    grid.fit(X,Y)

    return grid.best_estimator_



reg = fit_model1(X_train,y_train)


print("Parameter 'fgfg' is {} for the optimal model.".format(reg.get_params()))

predict_test = reg.predict(X_test)

acc = performance_metric(y_test,predict_test)

print("acc = {:.3f}".format(acc))


