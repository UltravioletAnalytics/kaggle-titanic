""" 
Kaggle Titanic competition

Adapted from myfirstforest.py comitted by @AstroDave 
"""
import loaddata
import learningcurve
import scorereport

import pandas as pd
import random as rd
import numpy as np
import time
import csv as csv
import sys
from sklearn import metrics
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import SGDClassifier
from operator import itemgetter



# Script
###################################
if __name__ == '__main__':
    # Do all the feature engineering
    print "Generating initial training/test sets"
    input_df, submit_df = loaddata.getDataSets(raw=False, binary=True, bins=False, scaled=True)
    
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
    
    # Remove ~100 records from the training data as a labeled test set (held out from validation)
    test_set_pct = 0.25
    rows = rd.sample(input_df.index, int(np.rint(input_df.shape[0]*test_set_pct)))
    X_test   = input_df.ix[rows]
    X_train  = input_df.drop(rows)
    print "Total number of training examples:", input_df.shape[0]
    print "Number of examples for training/validation: ", X_train.shape[0]
    print "Number of examples for testing: ", X_test.shape[0]
    
    
    # loop variables
    submission = []
    correct = 0.0
    
    
    train_data = X_train.values
    X = train_data[:, 1::]
    y = train_data[:, 0]
    
    submit_ids = submit_df['PassengerId'].values
    submit_df.drop('PassengerId', axis=1, inplace=True)
    submit_data = submit_df.values
    
    
    # specify model parameters and distributions to sample from
    n_iter = np.ceil(10**6 / X.shape[0])
    
    hinge_params = {"loss": ["hinge"],
                    "n_iter": [n_iter],
                    "alpha": [0.0001, 0.00001],
                    "penalty": ["elasticnet"],
                    "l1_ratio": 0.2*np.arange(0,5),
                    "shuffle": [True],
                    "learning_rate": ['optimal'],
                    "class_weight": ["auto"] }
    
    modified_huber_params = {"loss": ["modified_huber"],
              "n_iter": [n_iter],
              "alpha": [0.01, 0.001],
              "penalty": ["elasticnet"],
              "l1_ratio": 0.1*np.arange(0,10),
              "shuffle": [True],
              "learning_rate": ['optimal'] }
    
    log_params = {"loss": ["log"],
              "n_iter": [n_iter],
              "alpha": 10.0**-np.arange(4,5),
              "penalty": ["elasticnet"],
              "l1_ratio": 0.2*np.arange(0,5),
              "shuffle": [True],
              "learning_rate": ['constant','optimal', 'invscaling'],
              "eta0": 0.02*np.arange(0,4)+.01,
              "class_weight": ["auto"] }
    
    perceptron_params = {"loss": ["perceptron"],
              "n_iter": [n_iter],
              "alpha": 10.0**-np.arange(2,6),
              "penalty": ["elasticnet"],
              "l1_ratio": 0.2*np.arange(0,5),
              "shuffle": [True],
              "learning_rate": ['invscaling', 'constant', 'optimal'],
              "eta0": [0.001,0.005,0.01,0.05,0.1],
              "power_t": [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1] }
    
    params = { "loss": "hinge",
               "n_iter": n_iter,
               "alpha": 0.01,
               "penalty": "elasticnet",
               "l1_ratio": 0.,
               "shuffle": True,
               "learning_rate": 'optimal' }
     
    sgd = SGDClassifier()
    
    #==============================================================================================================
    # print 'Hyperparameter optimization via RandomizedSearchCV...'
    # n_iter_search = 12
    # random_search = RandomizedSearchCV(sgd, param_distributions=hinge_params, cv=4, n_iter=n_iter_search, n_jobs=-1)
    # random_search.fit(X, y)
    # best_params = scorereport.report(random_search.grid_scores_)
    #==============================================================================================================

    print 'Hyperparameter optimization via GridSearchCV...'
    grid_search = GridSearchCV(sgd, hinge_params, cv=3, n_jobs=-1)
    grid_search.fit(X, y)
    best_params = scorereport.report(grid_search.grid_scores_)
    
    
    # Use parameters from either the hyperparameter optimization, or manually selected parameters...
    params = best_params
    
    print "Generating SGDClassifier model with parameters: ", params
    sgd = SGDClassifier(**params)
    
    print 'Plot learning curve...'
    cv = ShuffleSplit(X.shape[0], n_iter=25, test_size=0.2, 
                      random_state=np.random.randint(0,123456789))
    title = "SGDClassifier: ", params
    learningcurve.plot_learning_curve(sgd, title, X, y, ylim=(0.5, 1.0), cv=cv, n_jobs=-1)
    
    
    test_data = X_test.values
    Xt = test_data[:, 1::]
    yt = test_data[:, 0]
    
    print "Training model with", train_data.shape[0], "examples"
    print "Testing model with", test_data.shape[0], "examples"
    print "Submitting predicted labels for", submit_df.shape[0], "records"
    
    test_scores = []
    # Using the optimal parameters, predict the survival of the labeled test set
    for i in range(5):
        print "Predicting test set for submission..."
        sgd.fit(X, y)
        print "train set score     :", sgd.score(X, y)
        print "test set score      :", sgd.score(Xt, yt)
        test_scores.append(sgd.score(Xt, yt))
    
    print "Mean - Std:", np.mean(test_scores)-np.std(test_scores)
    print "Examples in this test set:", Xt.shape[0]
    #print "correctly identified test examples in this gender:", (np.mean(test_scores) - np.std(test_scores)) * Xt.shape[0]
    correct += (np.mean(test_scores) - np.std(test_scores)) * Xt.shape[0]
    
    # Concatenate this model's predictions to the submission array
    passengerPredictions = zip(submit_ids, sgd.predict(submit_data))
    if len(submission) == 0:
        submission = np.asarray(passengerPredictions)
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
    predictions_file = open("data/results/" + oob + "sgd-hinge_" + str(int(time.time())) + ".csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(output)
    predictions_file.close()
    print 'Done.'
