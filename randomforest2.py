""" 
RandomForest 
"""
import loaddata
import scorereport
import learningcurve
import cProfile
import pandas as pd
import numpy as np
import time
import csv
import sys
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def scoreForest(estimator, X, y):
    score = estimator.oob_score_
    print "oob_score_:", score
    return score

# Script
###################################
if __name__ == '__main__':
    # Do all the feature engineering
    print "Generating initial training/test sets"
    input_df, submit_df = loaddata.getDataSets(bins=True, binary=True)
    
    pd.set_option('display.max_columns', None)
    print input_df.head()
        
    # Collect the test data's PassengerIds then drop it from the train and test sets
    submit_ids = submit_df['PassengerId']
    
    #==============================================================================================================
    # # Random Forests require numeric values (wants floats but can handle ints) and is scale invariant
    # droplist = ['PassengerId', 'Cabin', 'Name', 'Sex', 'Ticket', 'CabinNumber', 'Age_(21, 28]', 'Age_(36.5, 80]',
    #             'Age_(28, 36.5]', 'Age_[0.17, 21]', 'Title', 'Pclass', 'Embarked',
    #             #'Pclass_1', 'Pclass_2', 'Pclass_3',
    #             #'Embarked_83', 'Embarked_67', 'Embarked_81', 
    #             'Title_Lady', 'Title_Dr', 'Title_Sir', 'Title_Rev', 'Title_Master',
    #             'CabinLetter_65','CabinLetter_70','CabinLetter_68','CabinLetter_69','CabinLetter_67','CabinLetter_66']
    #             
    # input_df.drop(droplist, axis=1, inplace=1) 
    # submit_df.drop(droplist, axis=1, inplace=1)
    #==============================================================================================================
    
    print 'Training with', input_df.columns.size, 'features:', input_df.columns.values
    
    features_list = input_df.columns.values[1::] # Save for feature importance graph
    X = input_df.values[:, 1::]
    y = input_df.values[:, 0]
    
    sqrtfeat = np.sqrt(X.shape[1])
    
    # specify model parameters and distributions to sample from
    params_test = { "n_estimators"      : np.rint(np.linspace(X.shape[0]*2, X.shape[0]*3, 3)).astype(int),
                    "max_features"      : np.rint(np.linspace(sqrtfeat, sqrtfeat*2, 3)).astype(int),
                    "min_samples_split" : np.rint(np.linspace(2, X.shape[0]/50, 4)).astype(int) }

    params_test = { "n_estimators"      : [2000, 3000],
                    "max_features"      : np.rint(np.linspace(sqrtfeat, sqrtfeat*2, 4)).astype(int),
                    "min_samples_split" : np.rint(np.linspace(X.shape[0]/200, X.shape[0]/50, 4)).astype(int) }
    
    #print params_test
    
    params_score = { "n_estimators"      : 2000,
                     "max_features"      : 7,
                     "min_samples_split" : 12 }
    
    # Initial RandomForest model without any parameter tuning. We'll use this as a simple method to
    # trim down the feature set
    forest = RandomForestClassifier(oob_score=True, n_estimators=5000)
    forest.fit(X, y)
    
    ###############################################################################
    # Plot feature importance
    feature_importance = forest.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, features_list[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.draw()
    
    plt.show()
    sys.exit()
    
    #==========================================================================================================
    # print "Hyperparameter optimization using RandomizedSearchCV..."
    # rand_search = RandomizedSearchCV(forest, params, n_jobs=-1, n_iter=20)
    # rand_search.fit(X, y)
    # best_params = report(rand_search.grid_scores_)
    #==========================================================================================================
    
    print "Hyperparameter optimization using GridSearchCV..."
    cv = cross_validation.LeaveOneOut(X.shape[0])
    fparams = { 'sample_weight': np.array([1-np.mean(y) if s == 1 else np.mean(y) for s in y])}
    grid_search = GridSearchCV(forest, params_test, scoring=scoreForest, n_jobs=-1, cv=10, fit_params=fparams)
    grid_search.fit(X, y)
    #cProfile.run('grid_search.fit(X, y)')
    best_params = scorereport.report(grid_search.grid_scores_)


    # Use parameters from either the hyperparameter optimization, or manually selected parameters...
    params = best_params
    
    
    print "Generating RandomForestClassifier model with parameters: ", params
    forest = RandomForestClassifier(n_jobs=-1, oob_score=True, **params)
    
    print "Plot Learning Curve..."
    cv = cross_validation.KFold(n=X.shape[0], n_folds=10, shuffle=True)
    title = "RandomForestClassifier with hyperparams: ", params
    learningcurve.plot_learning_curve(forest, title, X, y, (0.6, 1.01), cv=cv, n_jobs=-1)
    #cProfile.run('learningcurve.plot_learning_curve(forest, title, X, y, (0.6, 1.01), cv=cv, n_jobs=-1)')
    plt.show()
    
    print "Submitting predicted labels for", submit_df.shape[0], "records"
    
    test_scores = []
    weights = np.array([1-np.mean(y) if s == 1 else np.mean(y) for s in y])
    # Using the optimal parameters, predict the survival of the labeled test set 10 times
    for i in range(10):
        print "Predicting round",i,"..."
        forest.fit(X, y, sample_weight=weights)
        #print "train set score     :", forest.score(X, y)
        print "train set oob_score :", forest.oob_score_
        test_scores.append(forest.oob_score_)
    
    print "OOB Mean:", np.mean(test_scores)
    print "Est. correctly identified test examples:", np.mean(test_scores) * X.shape[0]
    
    
    
    
    # build results
    submission = np.asarray(zip(submit_ids, forest.predict(submit_df)))
    
    accuracy = ("%.3f"%(np.mean(test_scores))).lstrip('0')
    print "**********************************"
    print "average oob accuracy:", accuracy
    print "**********************************"
    
    print "Submission shape: ", submission.shape
    submission = submission.astype(int)
    # sort so that the passenger IDs are back in the correct sequence
    output = submission[submission[:,0].argsort()]
    
    # write results
    predictions_file = open("data/results/" + accuracy + "randforest" + str(int(time.time())) + ".csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(output)
    
    plt.show()
    print 'Done.'