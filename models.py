import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from numpy import asarray
from sklearn.linear_model import ElasticNet

def random_forest_forecast(X_train, y_train, testX, n_trees):
    # transform list into array
    trainX = asarray(X_train)
    trainy = asarray(y_train)

    # fit model
    model = RandomForestRegressor(n_estimators=n_trees)
    model.fit(trainX, trainy)
    
    # make a one-step prediction
    yhat = model.predict([testX])
    return yhat[0]

def fit_random_forest_classifier(X_train, y_train, n_trees):
    # transform list into array
    trainX = asarray(X_train)
    trainy = asarray(y_train)

    # fit model
    model = RandomForestRegressor(n_estimators = n_trees)
    model.fit(trainX, trainy)
    
    return model 
    
def elasticnet_forecast(X_train, y_train, testX, alpha):
    # transform list into array
    trainX = asarray(X_train)
    trainy = asarray(y_train)
    
    # fit model
    model = ElasticNet(alpha = alpha)
    model.fit(trainX, trainy)

    # make a one-step prediction
    yhat = model.predict([testX])
    return yhat[0]

def fit_elasticnet_classifier(X_train, y_train, alpha):
    # transform list into array
    trainX = asarray(X_train)
    trainy = asarray(y_train)
    
    # fit model
    model = ElasticNet(alpha = alpha)
    model.fit(trainX, trainy)
    
    return model
