""" 
Kaggle Titanic competition

This file experiments with non-linear support vector machine classifiers
"""
import loaddata
import learningcurve

import sys
import numpy as np
import time
import csv as csv
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
from operator import itemgetter


# Globals
############################3



# Functions
############################
# Utility function to report optimal parameters
def report(grid_scores, n_top=5):
    params = None
    
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
        
        if params == None:
            params = score.parameters
    
    return params




# Script
###################################
if __name__ == '__main__':
        
    # Do all the feature engineering
    input_df, submit_df = loaddata.getDataSets(raw=False, binary=True, bins=False)
    submit_df.drop('Survived', axis=1, inplace=1)
    
    print 'All generated features: ' + str(list(input_df.columns.values))
    
    # Collect the test data's PassengerIds
    ids = submit_df['PassengerId'].values
    
    # Remove variables that aren't appropriate for this model:
    drop_list = ['PassengerId']
    input_df.drop(drop_list, axis=1, inplace=1) 
    submit_df.drop(drop_list, axis=1, inplace=1) 
    
    print 'Building SVC with ', len(input_df.columns), ' columns: ', list(input_df.columns.values)
    print "Number of training examples: ", input_df.shape[0]
    
    train_data = input_df.values
    X = train_data[0::,1::]
    y = train_data[0::,0]
    test_data = submit_df.values
    
    
    # specify model parameters and distributions to sample from
    rbf_params = {"kernel": ['rbf'],
                    "class_weight": ['auto'],
                    "C": [1],
                    "gamma": [0.1],
                    "tol": 10.0**-np.arange(2,4),
                    "random_state": [1234567890]}
    
    poly_params = {"kernel": ['poly'],
                    "class_weight": ['auto'],
                    "degree": [3],
                    "C": 10.0**np.arange(-1,1),
                    "gamma": 10.0**np.arange(-1, 1),
                    "coef0": 10.0**-np.arange(1,2),
                    "tol": 10.0**-np.arange(1,3),
                    "random_state": [1234567890]} # 4*9*7*5*3 = 3780 possible combinations
   
    sigmoid_params = {"kernel": ['sigmoid'],
                        "class_weight": ['auto'],
                        "C": 10.0**np.arange(-2,6),
                        "gamma": 10.0**np.arange(-3, 3),
                        "coef0": 10.0**-np.arange(1,5),
                        "tol": 10.0**-np.arange(2,4),
                        "random_state": [1234567890]}
    
    plot_params = {"kernel": 'poly',
                   "degree": 3,
                   "C": 1,
                   "gamma": 0.1,
                   "tol": .01,
                   "class_weight": 'auto',
                   "random_state": 1234567890}
    
    svc = SVC()
    
    print 'Hyperparameter optimization via RandomizedSearchCV...'
    i = 100
    random_search = RandomizedSearchCV(svc, param_distributions=poly_params, cv=10, n_iter=i, n_jobs=-1, verbose=2)
    random_search.fit(X, y)
    best_params = report(random_search.grid_scores_)

    sys.exit()

    #==============================================================================================================
    # print 'Hyperparameter optimization via GridSearchCV...'
    # grid_search = GridSearchCV(svc, rbf_params, cv=20, n_jobs=-1, verbose=2)
    # grid_search.fit(X, y)
    # best_params = report(grid_search.grid_scores_)
    #==============================================================================================================
    
    
    # Plot the learning curve for the model with the best parameters
    print 'Plotting learning curve...'
    cv = ShuffleSplit(X.shape[0], n_iter=20, test_size=0.33, random_state=np.random.randint(0,123456789))
    title = "SVC(RBF): ", best_params
    svc = SVC(**best_params)
    learningcurve.plot_learning_curve(svc, title, X, y, ylim=(0.5, 1.0), cv=cv, n_jobs=-1)
    
    sys.exit()
    
    # Using the optimal parameters, predict the survival of the test set
    print 'Predicting test set...'
    #==================================================================================================================
    # for train_ix, val_ix in cv:
    #     sgd.fit(X[train_ix], y[train_ix])
    #     val_pred = sgd.predict(X[val_ix])
    #     print "cross val accuracy score: ", metrics.accuracy_score(y[val_ix], val_pred)
    #==================================================================================================================
    svc.fit(X, y)
    output = svc.predict(test_data).astype(int)
    
     
    # write results
    predictions_file = open("data/results/svc-rbf_" + str(int(time.time())) + ".csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()
    print 'Done.'
