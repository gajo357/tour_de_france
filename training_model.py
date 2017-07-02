'''
Created on Jun 30, 2017

@author: Gajo
'''
from pro_cycling_scraper import rider_labels, stages_labels
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor, DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import LinearSVR, SVC

features = ['rider_id', 'rider_height', 'rider_weight', 'rider_one_day', 'rider_gc', 'rider_time_trial', 'rider_sprint', 'rider_pcs_rank', 'rider_uci_rank', 'stage_length', 'stage_type']
label = 'result'

def column_apply(row, source_df, label, prefix):
    #print(row)
    row_id = row[prefix + 'id']
    #print(id)
    index = None
    for i in range(0, len(source_df['id'])):
        if source_df['id'][i] == row_id:
            index = i
            break
    value = source_df[label][index]
    
    #series = pd.Series({column_name : value})    
    #print(value)
    return value

def create_columns(results_df, source_df, labels, prefix):
    for label in labels:
        if label == 'id':
            continue
        column_name = label
        if not label.startswith(prefix):
            column_name = prefix + label
        
        results_df[column_name] = results_df.apply(lambda row: column_apply(row, source_df, label, prefix), axis=1)     

def change_to_categorical():
    results_df = load_data()
    results_df['stage_type'].fillna('hills_flat', inplace=True)
    le = preprocessing.LabelEncoder()
    le.fit(results_df['stage_type'])
    print(le.classes_)
    
    results_df['stage_type'] = le.transform(results_df['stage_type'].values)
    results_df.to_csv('results_df1.csv', index=False)

def get_rider_cost(row, riders_cost_df):
    full_name = row['rider_name'].split()
    surname = full_name[0]
    names = full_name[1:]
    #print(id)
    matches = []
    matches_names = []
    for i in range(0, len(riders_cost_df['rider_name'])):
        fullname = riders_cost_df['rider_name'][i].split()
        surname_c = fullname[-1]
        names_c = fullname[:len(full_name) - 2]
        if surname == surname_c:
            matches.append((riders_cost_df['cost'][i], names_c))
        elif any(n in names for n in names_c):
            matches_names.append((riders_cost_df['cost'][i], surname_c))
    
    if len(matches) == 0:
        if len(matches_names) == 0:
            return 0
        if len(matches_names) == 1:
            return matches_names[0][0]
        for s in matches_names:
            if surname[0] == s[1][0]:
                return s[0]
        
        return 0
            
    if len(matches) == 1:
        return matches[0][0]
    
    for match in matches:
        if any(n in names for n in match[1]):
            return match[0]
        
    return matches[0][0]

def match_riders():
    riders_df = pd.read_csv('riders.csv')
    riders_cost_df = pd.read_csv('riders_cost.csv')
    riders_df['cost'] = riders_df.apply(lambda row: get_rider_cost(row, riders_cost_df), axis=1)
    
    riders_df.to_csv('riders_w_cost.csv', index=False)

def preprocess_data():
    riders_df = pd.read_csv('riders.csv')
    stages_df = pd.read_csv('stages.csv')
    results_df = pd.read_csv('short_results.csv')
        
    #print(results_df.head())
    create_columns(results_df, stages_df, stages_labels, 'stage_')
    #print(results_df.head())
    create_columns(results_df, riders_df, rider_labels, 'rider_')
    #print(results_df.head())
    
    results_df['stage_type'].fillna('flat', axis=0 , inplace=True)
    le = preprocessing.LabelEncoder()
    le.fit(results_df['stage_type'])
    print(le.classes_)    
    results_df['stage_type'] = le.transform(results_df['stage_type'].values)
    
    results_df['rider_pcs_rank'].fillna(1000, axis=0 , inplace=True)
    results_df['rider_uci_rank'].fillna(1000, axis=0 , inplace=True)
    results_df.fillna(results_df.mean(), inplace=True)
    
    results_df.to_csv('results_df.csv', index=False)
    
    return (riders_df, stages_df, results_df)

def load_data():
    data = pd.read_csv('results_df.csv')
    useful = list(features)
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
        
    reg = DecisionTreeRegressor(max_depth=40, min_samples_leaf=2, random_state=0)
    
    X = data.as_matrix(features)
    y = data[label].values
    
    print(cross_val_score(reg, X, y, cv = 10))
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

if __name__ == '__main__':
    train_model_Kn()