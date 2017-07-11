'''
Created on Jun 30, 2017

@author: Gajo
'''
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor, DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import LinearSVR, SVC

features = ['stage_length', 'stage_type']
label = 'time'

def load_data():
    data = pd.read_csv('results_df.csv')
    useful = list(features)
    useful.append('rider_id')
    useful.append(label)
    
    data = data[useful]
    data.dropna(axis=0, how='any', inplace=True)
    return data

def validate_model(model, X, y, features_train, labels_train, features_test, labels_test):
    shuffle = KFold(len(X), n_folds=5, shuffle=True, random_state=0)
    model.fit(features_train, labels_train)    
    print(model.score(features_train, labels_train))
    print(model.score(features_test, labels_test)) 
    print(cross_val_score(model, X, y, cv = shuffle))

def train_different():
    data = load_data()
    
    X = data.as_matrix(features)
    y = data[label].values
    ### split the data
    features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    
    
    print('Linear regression')
    model = LinearRegression()
    validate_model(model, X, y, features_train, labels_train, features_test, labels_test)
    
    print('DTR')
    model = DecisionTreeRegressor(max_depth=100, min_samples_leaf=2, random_state=0)
    validate_model(model, X, y, features_train, labels_train, features_test, labels_test)
    
    print('ETR')
    model = ExtraTreesRegressor(n_estimators=10, random_state=0)
    validate_model(model, X, y, features_train, labels_train, features_test, labels_test)
    
    print('K Neighbors')
    model = KNeighborsRegressor(n_neighbors=100, weights='uniform', algorithm='auto', leaf_size=30, n_jobs=-1)
    validate_model(model, X, y, features_train, labels_train, features_test, labels_test)
    
    print('Ridge')
    model = RidgeCV()
    validate_model(model, X, y, features_train, labels_train, features_test, labels_test)
    
    
    print('Linear SVR')
    model = LinearSVR(epsilon=0, C=10.0, random_state=0)
    validate_model(model, X, y, features_train, labels_train, features_test, labels_test)
    
def train_different_clf():
    data = load_data()
    
    X = data.as_matrix(features)
    y = data[label].values
    ### split the data
    features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    
    
    print('NB')
    model = GaussianNB()
    validate_model(model, X, y, features_train, labels_train, features_test, labels_test)
    
    print('DTR')
    model = DecisionTreeClassifier()
    validate_model(model, X, y, features_train, labels_train, features_test, labels_test)
    
    print('ETR')
    model = ExtraTreeClassifier()
    validate_model(model, X, y, features_train, labels_train, features_test, labels_test)
    
    print('K Neighbors')
    model = KNeighborsClassifier()
    validate_model(model, X, y, features_train, labels_train, features_test, labels_test)
      
    print('SVC')
    model = SVC()
    validate_model(model, X, y, features_train, labels_train, features_test, labels_test)
    
def train_model_dtr():
    data = load_data()
    riders = pd.read_csv('riders_w_cost.csv')
    
    for rider_id in riders['id']:        
        reg = DecisionTreeRegressor()
        rider_df = data[data['rider_id'] == rider_id]
        
        X = rider_df.as_matrix(features)
        y = rider_df[label].values
        
        features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
        reg.fit(features_train, labels_train) 
        
        joblib.dump(reg, 'models/model.pkl')
        
        print(reg.score(features_train, labels_train))
        print(reg.score(features_test, labels_test))
        
    
    pass

def train_model_linreg():
    data = load_data()
    reg = LinearRegression()
    
    X = data.as_matrix(features)
    y = data[label].values
    
    ### split the data
    features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    reg.fit(features_train, labels_train)
    
    print(reg.coef_)
    print(reg.intercept_)
    
    print(reg.score(features_train, labels_train))
    print(reg.score(features_test, labels_test))

    print(cross_val_score(reg, X, y, cv = 10))
    
    joblib.dump(reg, 'model.pkl')
    pass

def train_model_svm():
    data = load_data()
        
    reg = LinearSVR(epsilon=0, C=1.0, fit_intercept=True, random_state=42, max_iter=1000)
    
    X = data.as_matrix(features)
    y = data[label].values
    
    print(cross_val_score(reg, X, y, cv = 10))
    pass

def train_model_nb():
    data = load_data()
        
    reg = GaussianNB()
    
    X = data.as_matrix(features)
    y = data[label].values
    
    print(cross_val_score(reg, X, y, cv = 10))
    pass
    
def train_model_Kn():
    data = load_data()
    #reg = KNeighborsRegressor(n_neighbors=200, weights='uniform', algorithm='auto', leaf_size=30, n_jobs=-1)
    
    X = data.as_matrix(features)
    y = data[label].values
    
    ### split the data
    features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    
    parameters = {'n_neighbors': [5, 100, 200, 300], 
                  'weights': ['distance', 'uniform'],
                  'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'leaf_size': [1, 10, 50]}
    
    reg = GridSearchCV(KNeighborsRegressor(), parameters)
    #validate_model(reg, X, y, features_train, labels_train, features_test, labels_test)
    #reg = KNeighborsRegressor(algorithm='auto', leaf_size=10, metric='minkowski',
    #                          metric_params=None, n_jobs=1, n_neighbors=200, p=2,
    #                          weights='distance')
    reg.fit(features_train, labels_train) 
    print(reg.score(features_train, labels_train))
    print(reg.score(features_test, labels_test)) 
    
    joblib.dump(reg, 'model.pkl')
    print(reg.best_estimator_)
    pass

def train_model_KnClf():
    data = load_data()
    #reg = KNeighborsRegressor(n_neighbors=200, weights='uniform', algorithm='auto', leaf_size=30, n_jobs=-1)
    
    X = data.as_matrix(features)
    y = data[label].values
    
    ### split the data
    features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    
    parameters = {'n_neighbors': [5, 100, 200], 
                  'weights': ['uniform'],
                  'algorithm' : ['auto'],
                  'leaf_size': [1]}
    
    reg = GridSearchCV(KNeighborsClassifier(), parameters)
    #validate_model(reg, X, y, features_train, labels_train, features_test, labels_test)
    reg.fit(features_train, labels_train) 
    predictions = reg.predict(features_test)
    print(reg.best_estimator_)
    print(reg.score(features_train, labels_train))
    print(reg.score(features_test, labels_test)) 
    print(confusion_matrix(labels_test, predictions))
    print(precision_score(labels_test, predictions, average=None))
    print(recall_score(labels_test, predictions, average=None))
    
    joblib.dump(reg, 'model.pkl')
    print(reg.best_estimator_)
    pass

def train_model(model_creator, save_model):
    data = load_data()
    riders = pd.read_csv('riders_w_cost.csv')
    
    for rider_id in riders['id']:
        print(rider_id)
        rider_df = data[data['rider_id'] == rider_id]
        if len(rider_df[label]) < 5:
            print()
            continue
        
        X = rider_df.as_matrix(features)
        y = rider_df[label].values
        
        features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        
        reg = model_creator()
        reg.fit(features_train, labels_train) 
        
        if save_model:
            joblib.dump(reg, 'models/model_{}.pkl'.format(rider_id))
        
        print(reg.score(features_train, labels_train))
        print(reg.score(features_test, labels_test))
        print()
        
    
    pass

if __name__ == '__main__':
    train_model(lambda: RidgeCV(), True)