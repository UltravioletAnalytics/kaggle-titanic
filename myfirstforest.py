""" 
Kaggle Titanic competition

Adapted from myfirstforest.py comitted by @AstroDave 
"""
import loaddata
import learningcurve

import numpy as np
import time
import csv
import sys
import re
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from operator import itemgetter


# Globals
############################3
ports_dict = {}               # Holds the possible values of 'Embarked' variable

cabinletter_matcher = re.compile("([a-zA-Z]+)")
cabinnumber_matcher = re.compile("([0-9]+)")



# Functions
############################
# Utility function to report optimal parameters
def report(grid_scores, n_top=18):
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
    print "Generating initial training/test sets"
    input_df, test_df = loaddata.getDataSets(raw=False, binary=True, bins=False, scaled=True)
    
    # Collect the test data's PassengerIds then drop it from the train and test sets
    ids = test_df['PassengerId'].values
    drop_list = ['PassengerId']
    input_df.drop(drop_list, axis=1, inplace=1) 
    test_df.drop(drop_list, axis=1, inplace=1) 
    
    
    # Run dimensionality reduction and clustering on the remaining feature set. This will return an unlabeled
    # set of derived parameters along with the ClusterID so we can train multiple models for different groups
    print "Final feature set before Reduction and Clustering: ", input_df.columns.values
    print "Dimensionality Reduction and Clustering..."
    input_df, test_df = loaddata.reduceAndCluster(input_df, test_df)
    
    
    print 'Building RandomForestClassifier with ' + str(len(input_df.columns)) \
          + ' columns: ' + str(list(input_df.columns.values))
    
    print "Number of training examples: " + str(input_df.shape[0])
    
    
    print "*************Build X_train, X_test, and submission_data"
    train_data = input_df.values
    X = train_data[:, 1::]
    y = train_data[:, 0]
    X_train, X_test, y_train, y_test \
        = cross_validation.train_test_split(X, y, test_size=0.2, random_state=np.random.randint(0,123456789))
    submission_data = test_df.values
    
    
    # specify model parameters and distributions to sample from
    params = {"n_estimators": [2000, 4000],
              "max_depth": [3,4,5],
              "max_features": np.arange(3, int(np.round(np.sqrt([X_train.shape[1]])))),
              "bootstrap": [True],
              "oob_score": [True]}
    
    plot_params = {"n_estimators": 2000,
                   "max_depth": 3,
                   "max_features": 4,
                   "bootstrap": True,
                   "oob_score": True}
    
    forest = RandomForestClassifier()
    
    #==============================================================================================================
    # print "Hyperparameter optimization using RandomizedSearchCV..."
    # rand_search = RandomizedSearchCV(forest, params, n_jobs=-1, n_iter=20)
    # rand_search.fit(X_train, y_train)
    # best_params = report(rand_search.grid_scores_)
    #==============================================================================================================
    
    #==============================================================================================================
    # print "Hyperparameter optimization using GridSearchCV..."
    # grid_search = GridSearchCV(forest, params, n_jobs=-1)
    # grid_search.fit(X_train, y_train)
    # best_params = report(grid_search.grid_scores_)
    #==============================================================================================================
    
    print "Plot Learning Curve..."
    cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=3, test_size=0.25, \
                                       random_state=np.random.randint(0,123456789))
    title = "RandomForestClassifier with hyperparams: ", plot_params
    forest = RandomForestClassifier(n_jobs=-1, **plot_params)
    learningcurve.plot_learning_curve(forest, title, X_train, y_train, (0.6, 1.01), cv=cv, n_jobs=-1)
    
    
    # Using the optimal parameters, predict the survival of the test set
    print "Predicting test set for submission..."
    forest.fit(X_train, y_train)
    print "train set oob_score: ", forest.oob_score_
    print "test set accuracy score: ", forest.score(X_test, y_test)
    output = forest.predict(submission_data).astype(int)
    
     
    # write results
    predictions_file = open("data/results/randforest" + str(int(time.time())) + ".csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()
    print 'Done.'
