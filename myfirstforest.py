""" 
Kaggle Titanic competition

Adapted from myfirstforest.py comitted by @AstroDave 
"""
import loaddata
import learningcurve
import scorereport

import random as rd
import pandas as pd
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


# Script
###################################
if __name__ == '__main__':
    # Do all the feature engineering
    print "Generating initial training/test sets"
    input_df, submit_df = loaddata.getDataSets(raw=False, binary=True, bins=False, scaled=True, balanced=True)
    
    # Collect the test data's PassengerIds then drop it from the train and test sets
    submit_ids = submit_df['PassengerId']
    input_df.drop(['PassengerId'], axis=1, inplace=1) 
    submit_df.drop(['PassengerId'], axis=1, inplace=1) 
    
    # Run dimensionality reduction and clustering on the remaining feature set. This will return an unlabeled
    # set of derived parameters along with the ClusterID so we can train multiple models for different groups
    print "Dimensionality Reduction and Clustering..."
    input_df, submit_df = loaddata.reduceAndCluster(input_df, submit_df, 2)
    
    # Add the passenger ID back into the test set so we can keep track of them as we train different models
    submit_df = pd.concat([submit_ids, submit_df], axis=1)
    
    print 'Generated', input_df.columns.size, 'features:', input_df.columns.values
    
    # Remove 5% records from the training data as a labeled test set (held out from validation)
    test_set_pct = 0.05
    rows = rd.sample(input_df.index, int(np.rint(input_df.shape[0]*test_set_pct)))
    X_test   = input_df.ix[rows]
    X_train  = input_df.drop(rows)
    print "Total number of training examples:", input_df.shape[0]
    print "Number of examples for training/validation: ", X_train.shape[0]
    print "Number of examples for testing: ", X_test.shape[0]
    
    # loop variables
    submission = []
    correct = 0.0
    
    # build models for each gender
    for gender in np.unique(input_df.Gender):
        
        print "*************************************************************"
        print "Processing gender: ", 'male' if gender == 1 else 'female'
        
        # Get all training examples for this gender
        train_data = X_train[X_train.Gender==gender].drop('Gender', axis=1).values
        X = train_data[:, 1::]
        y = train_data[:, 0]
        
        print submit_df.columns.values
        submit_gender_df = submit_df[submit_df.Gender==gender].drop('Gender', axis=1)
        submit_ids = submit_gender_df['PassengerId'].values
        submit_gender_df.drop('PassengerId', axis=1, inplace=True)
        submit_data = submit_gender_df.values
        
        # specify model parameters and distributions to sample from
        params = {"n_estimators": [1000,2000],
                  "max_depth": [3,4],
                  "max_features": [3,4,5],
                  "bootstrap": [True],
                  "oob_score": [True]}
        
        plot_params = {"n_estimators": 1000,
                       "max_depth": 3,
                       "max_features": 4,
                       "bootstrap": True,
                       "oob_score": True}
        
        forest = RandomForestClassifier()
    
        #==========================================================================================================
        # print "Hyperparameter optimization using RandomizedSearchCV..."
        # rand_search = RandomizedSearchCV(forest, params, n_jobs=-1, n_iter=20)
        # rand_search.fit(X, y)
        # best_params = report(rand_search.grid_scores_)
        #==========================================================================================================
        
        print "Hyperparameter optimization using GridSearchCV..."
        grid_search = GridSearchCV(forest, params, n_jobs=-1)
        grid_search.fit(X, y)
        best_params = scorereport.report(grid_search.grid_scores_)
    
        # Use parameters from either the hyperparameter optimization, or manually selected parameters...
        params = best_params
        
        
        print "Generating RandomForestClassifier model with parameters: ", params
        forest = RandomForestClassifier(n_jobs=-1, **params)
        
        print "Plot Learning Curve..."
        cv = cross_validation.ShuffleSplit(train_data.shape[0], n_iter=5, test_size=0.25, \
                                           random_state=np.random.randint(0,123456789))
        title = "RandomForestClassifier with hyperparams: ", params
        learningcurve.plot_learning_curve(forest, title, X, y, (0.6, 1.01), cv=cv, n_jobs=-1)
        
        
        test_data = X_test[X_test.Gender==gender].drop('Gender', axis=1).values
        Xt = test_data[:, 1::]
        yt = test_data[:, 0]
        
        print "Training", gender, "model with", train_data.shape[0], "examples"
        print "Testing", gender, "model with", test_data.shape[0], "examples"
        print "Submitting predicted labels for", submit_df.shape[0], "records"
        
        test_scores = []
        # Using the optimal parameters, predict the survival of the labeled test set
        for i in range(5):
            print "Predicting test set for submission..."
            forest.fit(X, y)
            print "train set score     :", forest.score(X, y)
            print "train set oob_score :", forest.oob_score_
            print "test set score      :", forest.score(Xt, yt)
            test_scores.append(forest.score(Xt, yt))
        
        print "Mean - Std:", np.mean(test_scores)-np.std(test_scores)
        print "Examples in this gender's test set:", Xt.shape[0]
        print "correctly identified test examples in this gender:", (np.mean(test_scores) - np.std(test_scores)) * Xt.shape[0]
        correct += (np.mean(test_scores) - np.std(test_scores)) * Xt.shape[0]
        
        # Concatenate this model's predictions to the submission array
        passengerPredictions = zip(submit_ids, forest.predict(submit_data))
        if len(submission) == 0:
            submission = passengerPredictions
        else:
            submission = np.concatenate([submission, passengerPredictions])
    
    oob = ("%.3f"%(correct/X_test.shape[0])).lstrip('0')
    print "**********************************"
    print "total test set oob:", oob
    print "**********************************"
    
    print "Submission shape: ", submission.shape
    submission = submission.astype(int)
    # sort so that the passenger IDs are back in the correct sequence
    output = submission[submission[:,0].argsort()]
    
    # write results
    predictions_file = open("data/results/" + oob + "randforest" + str(int(time.time())) + ".csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(output)
    #predictions_file.close()
    print 'Done.'
