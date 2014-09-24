""" 
Kaggle Titanic competition - Naive Bayes model tests

Adapted from myfirstforest.py comitted by @AstroDave 
"""
import loaddata
import pandas as pd
import numpy as np
import time
import csv as csv
import sys
import sklearn.cross_validation
import learningcurve
from sklearn import naive_bayes
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from operator import itemgetter



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

input_df, submit_df = loaddata.getDataSets(bins=False, scaled=True, raw=False)

# Collect the test data's PassengerIds
ids = submit_df['PassengerId'].values

# Remove variables that we couldn't transform into features: 
drop_list = ['PassengerId']
input_df.drop(drop_list, axis=1, inplace=1) 
submit_df.drop(drop_list, axis=1, inplace=1) 
submit_df.drop('Survived', axis=1, inplace=1)

print 'Building Naive Bayes Classifier with ' + str(len(input_df.columns)) \
      + ' columns: ' + str(list(input_df.columns.values))

train_data = input_df.values
X = train_data[0::,1::]
y = train_data[0::,0]
test_data = submit_df.values

#==================================================================================================================
# # specify model parameters and distributions to sample from
# params = {"alpha": np.random.rand(),
#           "binarize": np.random.rand()}
# # run randomized search to find the optimal parameters
# n_iter_search = 50
# bnb = naive_bayes.BernoulliNB()
# random_search = RandomizedSearchCV(bnb, param_distributions=params, n_iter=n_iter_search)
# random_search.fit(train_data[0::,1::], train_data[0::,0])
# best_params = report(random_search.grid_scores_)
#==================================================================================================================


# Plot the learning curve for the model
cv = sklearn.cross_validation.ShuffleSplit(X.shape[0], n_iter=100, train_size=0.7, test_size=0.3, 
                                           random_state=np.random.randint(0,123456789))
title = "Learning Curves (GaussianNB)"
gnb = naive_bayes.GaussianNB()
learningcurve.plot_learning_curve(gnb, title, X, y, (0.6, 0.9), cv=cv, n_jobs=1)


# Using the optimal parameters, predict the survival of the test set
print 'Predicting...'
bnb = naive_bayes.BernoulliNB()
bnb.fit(train_data[0::,1::], train_data[0::,0])
output = bnb.predict(test_data).astype(int)
  
# write results
predictions_file = open("data/results/naivebayes_gaussian" + str(int(time.time())) + ".csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
