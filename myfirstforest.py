""" 
Kaggle Titanic competition

Adapted from myfirstforest.py comitted by @AstroDave 
"""
import loaddata

import pandas as pd
import numpy as np
import math
import time
import csv
import sys
import re
import learningcurve
import sklearn.cross_validation
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from operator import itemgetter


# Globals
############################3
ports_dict = {}               # Holds the possible values of 'Embarked' variable

cabinletter_matcher = re.compile("([a-zA-Z]+)")
cabinnumber_matcher = re.compile("([0-9]+)")



# Functions
############################
# Utility function to report optimal parameters
def report(grid_scores, n_top=3):
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

# Do all the feature engineering
input_df, test_df = loaddata.getDataSets(binary=False, bins=False, scaled=False)
test_df.drop('Survived', axis=1, inplace=1)

# Collect the test data's PassengerIds
ids = test_df['PassengerId'].values

# Remove variables that aren't appropriate for this model:
drop_list = ['Name', 'Sex', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'PassengerId']
input_df.drop(drop_list, axis=1, inplace=1) 
test_df.drop(drop_list, axis=1, inplace=1) 

print 'Building RandomForestClassifier with ' + str(len(input_df.columns)) \
      + ' columns: ' + str(list(input_df.columns.values))

print "Number of training examples: " + str(input_df.shape[0])

train_data = input_df.values
X = train_data[0::,1::]
y = train_data[0::,0]
test_data = test_df.values


#==================================================================================================================
# # specify model parameters and distributions to sample from
# params = {"n_estimators": sp_randint(20, 100),
#           "max_depth": [3, None],
#           "max_features": sp_randint(1, train_data.shape[1] - 1),
#           "min_samples_split": sp_randint(1, int(train_data.shape[0] / 10)),
#           "min_samples_leaf": sp_randint(1, int(train_data.shape[0] / 20)),
#           "bootstrap": [True, False],
#           "criterion": ["gini", "entropy"]}
# 
# # run randomized search to find the optimal parameters
# n_iter_search = 500
# forest = RandomForestClassifier()
# random_search = RandomizedSearchCV(forest, param_distributions=params, n_iter=n_iter_search)
# random_search.fit(train_data[0::,1::], train_data[0::,0])
# best_params = report(random_search.grid_scores_)
#==================================================================================================================

cv = sklearn.cross_validation.ShuffleSplit(X.shape[0], n_iter=10, train_size=0.8, test_size=0.2, 
                                           random_state=np.random.randint(0,123456789))

trees=100
depth=round(math.ceil((len(input_df.columns)-1)/2.0))

title = "RandomForestClassifier: n_estimators=" + str(trees) + ", max_depth=" + str(depth)
forest = RandomForestClassifier(n_estimators=trees, max_depth=depth, oob_score=True)
learningcurve.plot_learning_curve(forest, title, X, y, (0.6, 1.01), cv=cv, n_jobs=1)


# Using the optimal parameters, predict the survival of the test set
print "Predicting with n_estimators=" + str(trees) + ", max_depth=" + str(depth)
forest = RandomForestClassifier(n_estimators=trees, max_depth=depth, oob_score=True)
forest.fit(X, y)
print forest.oob_score_
output = forest.predict(test_data).astype(int)

 
# write results
predictions_file = open("data/results/randforest" + str(int(time.time())) + ".csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
