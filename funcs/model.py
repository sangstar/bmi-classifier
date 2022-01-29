import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import json

import warnings
warnings.filterwarnings("ignore")

bmi_dict = {
    0:'Extremely weak',
    1:'Weak',
    2:'Normal',
    3:'Overweight',
    4:'Obesity',
    5: 'Extreme Obesity'
}

def get_best_params(x_train, y_train):

    params_grid= {
                    'max_depth' : [4, 6, 8,12],
                    'n_estimators': [50, 10, 100],
                    'max_features': ['sqrt', 'auto', 'log2'],
                    'min_samples_split': [2, 3, 10, 20],
                    'min_samples_leaf': [1, 3, 10, 20],
                    'bootstrap': [True, False],
    }


    rfc = RandomForestClassifier()
    gridsearch = GridSearchCV(
        estimator = rfc, 
        scoring = 'accuracy', 
        param_grid = params_grid, 
        cv = StratifiedKFold(n_splits=5), 
        n_jobs = -1, 
        )
    gridsearch.fit(x_train, y_train)
    best_params = gridsearch.best_params_
    with open('files/best_params.json', 'w') as f:
        json.dump(best_params, f)
    return best_params

def get_dat():
    df = pd.read_csv('files/bmi.csv')
    df = df.sample(frac=1)
    y = df['Index']
    df = df.drop(columns = ['Index'])
    df = pd.get_dummies(df, columns = ['Gender'])
    x = df
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    return x_train, x_test, y_train, y_test


def get_model():
    x_train, x_test, y_train, y_test = get_dat()
    try:
        with open('files/best_params.json', 'r') as f:
            params = json.load(f)
    except FileNotFoundError:
        print('Setting up model...')
        params = get_best_params(x_train, y_train)
    print('Model built.')
    rfc = RandomForestClassifier(**params)
    rfc.fit(x_train, y_train)
    print('Model has an accuracy of ', rfc.score(x_test, y_test)*100, 'percent on the test set.')
    return rfc

def get_bmi():
    model = get_model()
    height = input('Please enter height (cm). ')
    weight = input('Please enter weight (kg). ')
    sex = input('Male or female? ')
    return assess_bmi(height, weight, sex, model)

def assess_bmi(height, weight, sex, model):
    if sex.lower() == 'female':
        female = 1
    else:
        female = 0
    if sex.lower() == 'male':
        male = 1
    else:
        male = 0
    bmi = model.predict(np.array([[height, weight, female, male]]))[0]
    return print('For weight',weight, 'kg and height',height, 'cm and gender', sex, 'your classification is', bmi_dict[bmi])


