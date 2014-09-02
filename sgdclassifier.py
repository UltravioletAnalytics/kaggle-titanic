""" 
Kaggle Titanic competition

Adapted from myfirstforest.py comitted by @AstroDave 
"""
import loaddata
import learningcurve

import numpy as np
import time
import csv as csv
import sys
from sklearn import metrics
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import SGDClassifier
from operator import itemgetter


# Globals
############################



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
input_df, test_df = loaddata.getDataSets(raw=False, binary=True, bins=False)
test_df.drop('Survived', axis=1, inplace=1)

print 'All generated features: ' + str(list(input_df.columns.values))

# Collect the test data's PassengerIds
ids = test_df['PassengerId'].values

# Remove variables that aren't appropriate for this model:
drop_list = ['PassengerId']
input_df.drop(drop_list, axis=1, inplace=1) 
test_df.drop(drop_list, axis=1, inplace=1) 

print 'Building SGDClassifier with ' + str(len(input_df.columns)) \
      + ' columns: ' + str(list(input_df.columns.values))

print "Number of training examples: " + str(input_df.shape[0])

train_data = input_df.values
X = train_data[0::,1::]
y = train_data[0::,0]
test_data = test_df.values

n_iter = np.ceil(10**6 / X.shape[0])

hinge_params = {"loss": ["hinge"],
                "n_iter": [n_iter],
                "alpha": [0.0001, 0.00001],
                "penalty": ["elasticnet"],
                "l1_ratio": [0, 0.2, 0.4, 0.6, 0.8, 1],
                "shuffle": [True],
                "learning_rate": ['optimal']
                }

log_params = {"loss": ["hinge", "log", "modified_huber", "perceptron", "huber", "epsilon_insensitive"],
          "alpha": 10.0**-np.arange(1,7),
          "penalty": ["l1", "l2", "elasticnet"],
          "n_iter": [n_iter],
          "shuffle": [True],
          "learning_rate": ['constant', 'optimal', 'invscaling'],
          "eta0": [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1],
          "power_t": [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]
          }

modified_huber_params = {"loss": ["hinge", "log", "modified_huber", "perceptron", "huber", "epsilon_insensitive"],
          "alpha": 10.0**-np.arange(1,7),
          "penalty": ["l1", "l2", "elasticnet"],
          "n_iter": [n_iter],
          "shuffle": [True],
          "learning_rate": ['constant', 'optimal', 'invscaling'],
          "eta0": [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1],
          "power_t": [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]
          }

# specify model parameters and distributions to sample from
params = {"loss": "hinge",
            "n_iter": n_iter,
            "alpha": 0.0001,
            "penalty": "elasticnet",
            "l1_ratio": 0.2,
            "shuffle": True,
            "learning_rate": 'optimal'
            }

 
sgd = SGDClassifier()

#==================================================================================================================
# #==================================================================================================================
# # #run randomized search to find the optimal parameters
# # n_iter_search = 12
# # random_search = RandomizedSearchCV(sgd, param_distributions=hinge_params, n_iter=n_iter_search)
# # random_search.fit(X, y)
# # best_params = report(random_search.grid_scores_)
# #==================================================================================================================
# grid_search = GridSearchCV(sgd, hinge_params, cv=5)
# grid_search.fit(X, y)
# best_params = report(grid_search.grid_scores_)
#==================================================================================================================

# Plot the learning curve for the model with the best parameters
cv = ShuffleSplit(X.shape[0], n_iter=2, test_size=0.25, 
                  random_state=np.random.randint(0,123456789))
title = "SGDClassifier" + str(params)
sgd = SGDClassifier(**params)
learningcurve.plot_learning_curve(sgd, title, X, y, ylim=(0.5, 1.0), cv=cv, n_jobs=1)


# Using the optimal parameters, predict the survival of the test set
print 'Predicting test set...'
for train_ix, val_ix in cv:
    sgd.fit(X[train_ix], y[train_ix])
    val_pred = sgd.predict(X[val_ix])
    print "cross val accuracy score: ", metrics.accuracy_score(y[val_ix], val_pred)

sgd.fit(X, y)
output = sgd.predict(test_data).astype(int)

 
# write results
predictions_file = open("data/results/sgdclassifier" + str(int(time.time())) + ".csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
