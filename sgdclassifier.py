""" 
Kaggle Titanic competition

Adapted from myfirstforest.py comitted by @AstroDave 
"""
import pandas as pd
import numpy as np
import time
import csv as csv
import re
from sklearn.grid_search import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from operator import itemgetter


# Globals
############################3
ports_dict = {}               # Holds the possible values of 'Embarked' variable

cabinletter_matcher = re.compile("([a-zA-Z]+)")
cabinnumber_matcher = re.compile("([0-9]+)")

ensemble_size = 5

# Functions
############################

### This method will transform the raw data into the features that the model will operate on
def munge(df):     
    # Gender => 'Female' = 0, 'Male' = 1
    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
 
    #==============================================================================================================
    # # Embarked => Three classes - 'C', 'Q', 'S'
    # df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values
    # ports_dict = getPorts(df)
    # for port in ports_dict.keys():
    #     df['Embarked_' + str(port)] = np.where(df['Embarked'] == port, 1, 0)
    #==============================================================================================================
    
    # Age => continuous value and missing values are replaced with median value
    median_age = df['Age'].dropna().median()
    if len(df.Age[ df.Age.isnull() ]) > 0:
        df.loc[ (df.Age.isnull()), 'Age'] = median_age
    # scale age between 1 and 0
    df['Age'] = df['Age'].map( lambda x: (x - df['Age'].min())/(df['Age'].max()-df['Age'].min()) )
    
    
    # missing familial counts are 0
    df['SibSp'] = df['SibSp'].fillna(0)
    df['Parch'] = df['Parch'].fillna(0)
    # scale SibSp and parch between 1 and 0
    df['SibSp'] = df['SibSp'].map( lambda x: (x - df['SibSp'].min())/float(df['SibSp'].max()-df['SibSp'].min()) )
    df['Parch'] = df['Parch'].map( lambda x: (x - df['Parch'].min())/float(df['Parch'].max()-df['Parch'].min()) )


    # Fare => if fare is missing, use the median for the passenger class
    if len(df.Fare[ df.Fare.isnull() ]) > 0:
        median_fare = np.zeros(3)
        for f in range(0,3):                                              # loop 0 to 2
            median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()
        for f in range(0,3):                                              # loop 0 to 2
            df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]
    # scale fare
    df['Fare'] = df['Fare'].map( lambda x: (x - df['Fare'].min())/float(df['Fare'].max()-df['Fare'].min()) )
    
    
    # Passenger Class => Three classes - 1, 2, and 3, replace missing values with mode
    df.Pclass[ df.Pclass.isnull() ] = df.Pclass.dropna().mode().values
    #==============================================================================================================
    # # convert to binary features
    # for pc in df.Pclass.unique():
    #     df['Pclass_' + str(pc)] = np.where(df['Pclass'] == pc, 1, 0)
    #==============================================================================================================
    # scale Pclass
    df['Pclass'] = df['Pclass'].map( lambda x: (x - df['Pclass'].min())/float(df['Pclass'].max()-df['Pclass'].min()) )


    #==============================================================================================================
    # # Cabin =>
    # df['Cabin'][df.Cabin.isnull()] = 'U0'
    # df['CabinLetter'] = df['Cabin'].map( lambda x : getCabinLetter(x))
    # df['CabinNumber'] = df['Cabin'].map( lambda x : getCabinNumber(x))
    # 
    # # scale CabinLetter and Number
    # df['CabinLetter'] = df['CabinLetter'].map( lambda x: (x - df['CabinLetter'].min())/float(df['CabinLetter'].max()-df['CabinLetter'].min()) )
    # df['CabinNumber'] = df['CabinNumber'].map( lambda x: (x - df['CabinNumber'].min())/float(df['CabinNumber'].max()-df['CabinNumber'].min()) )
    #==============================================================================================================
    
    return df



### This method will generate and/or return the dictionary of possible values of 'Embarked' => index for each value
def getPorts(df):
    global ports_dict
    
    if len(ports_dict) == 0:
        # determine distinct values of 'Embarked' variable
        ports = list(enumerate(np.unique(df['Embarked'])))
        # set the global dictionary
        ports_dict = { name : i for i, name in ports }
    
    return ports_dict

### Find the letter component of the cabin variable) 
def getCabinLetter(cabin):
    match = cabinletter_matcher.search(cabin)
    if match:
        return ord(match.group())
    else:
        return 'U'
 
### Find the number component of the cabin variable) 
def getCabinNumber(cabin):
    match = cabinnumber_matcher.search(cabin)
    if match:
        return float(match.group())
    else:
        return 0

# Utility function to report optimal parameters
def report(grid_scores, n_top=ensemble_size):
    params = []
    
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
        
        params.append(score.parameters)
    
    return params


# Script
###################################

# read in the training and testing data into Pandas.DataFrame objects
input_df = pd.read_csv('data/raw/train.csv', header=0)
test_df  = pd.read_csv('data/raw/test.csv',  header=0)

# data cleanup
input_df = munge(input_df)
test_df  = munge(test_df)

# Collect the test data's PassengerIds
ids = test_df['PassengerId'].values

# Remove variables that we couldn't transform into features:
input_df = input_df.drop(['Name', 'Sex', 'Embarked', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
test_df  = test_df.drop(['Name', 'Sex', 'Embarked', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 

print 'Building SGD Classifier with ' + str(len(input_df.columns)) \
      + ' columns: ' + str(list(input_df.columns.values))
    
print "Number of training examples: " + str(input_df.shape[0])

train_data = input_df.values
test_data = test_df.values

# specify model parameters and distributions to sample from
params = {"loss": ["log", "modified_huber", "perceptron", "huber", "epsilon_insensitive"], #"hinge", 
          "alpha": [0.0001,0.0005,0.001,0.005],
          "penalty": ["l1", "l2", "elasticnet"],
          "n_iter": [5],
          "shuffle": [False, True],
          "learning_rate": ['constant', 'optimal', 'invscaling'],
          "eta0": [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1],
          "power_t": [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]
          }
 
# run randomized search to find the optimal parameters
n_iter_search = 1000
sgd = SGDClassifier()
random_search = RandomizedSearchCV(sgd, param_distributions=params, n_iter=n_iter_search)
random_search.fit(train_data[0::,1::], train_data[0::,0])
best_params = report(random_search.grid_scores_)
 
  
# Using the optimal parameters, predict the survival of the test set
print 'Predicting...'
output = pd.Series(np.zeros(test_data.shape[0]).astype(int))
for bp in best_params:
    sgd = SGDClassifier(**bp).fit(train_data[0::,1::], train_data[0::,0])
    output += sgd.predict(test_data).astype(int)

output = output.map( lambda x: np.where(x >= ensemble_size/2, 1, 0) )

survivedPct = output.values.sum() / float(output.shape[0])
print survivedPct
print 1.0 - survivedPct


# write results
predictions_file = open("data/results/sgdclassifier" + str(int(time.time())) + ".csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
