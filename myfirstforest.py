""" 
Kaggle Titanic competition

Adapted from myfirstforest.py comitted by @AstroDave 
""" 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import csv as csv
#import re
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# Globals
############################3
ports_dict = {}               # Holds the possible values of 'Embarked' variable

#cabinletter_matcher = re.compile("([a-zA-Z]+)")
#cabinnumber_matcher = re.compile("([0-9]+)")



# Functions
############################

### This method will transform the raw data into the features that the model will operate on
def munge(df):     
    # Gender => 'Female' = 0, 'Male' = 1
    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
 
    # Embarked => Four classes - 'C', 'Q', 'S', and NULL (create a separate class for unknown)
    if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
        df.Embarked[ df.Embarked.isnull() ] = 'U'
 
    ports_dict = getPorts(df)
    df.Embarked = df.Embarked.map( lambda x: ports_dict[x]).astype(int)     # Convert all Embark strings to int


    # AgeClass => Six classes = Unknown(?), Baby(<3), Child(3-12), Teen(13-18), Adult(19-64), Senior(>65)
    df['AgeClass'] = df['Age'].map( lambda x : getAgeClass(x) )


    # Age => continuous value and missing values are replaced with median value
    median_age = df['Age'].dropna().median()
    if len(df.Age[ df.Age.isnull() ]) > 0:
        df.loc[ (df.Age.isnull()), 'Age'] = median_age


    # Fare => 
    if len(df.Fare[ df.Fare.isnull() ]) > 0:
        median_fare = np.zeros(3)
        for f in range(0,3):                                              # loop 0 to 2
            median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()
        for f in range(0,3):                                              # loop 0 to 2
            df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]
    
    # Cabin =>
    #df['Cabin'][df.Cabin.isnull()] = 'U0'
    #df['CabinLetter'] = df['Cabin'].map( lambda x : getCabinLetter(x))
    #df['CabinNumber'] = df['Cabin'].map( lambda x : getCabinNumber(x))
    
    return df


### Simple method to split passengers into typical age groups
def getAgeClass(age):
    if np.isnan(age):
        return 0
    elif age < 3:
        return 1
    elif age < 13:
        return 2
    elif age < 19:
        return 3
    elif age < 65:
        return 4
    else:
        return 5


### This method will generate and/or return the dictionary of possible values of 'Embarked' => index for each value
def getPorts(df):
    global ports_dict
    
    if len(ports_dict) == 0:
        # determine distinct values of 'Embarked' variable
        ports = list(enumerate(np.unique(df['Embarked'])))
        # set the global dictionary
        ports_dict = { name : i for i, name in ports }
    
    return ports_dict

#==================================================================================================================
# ### Find the letter component of the cabin variable) 
# def getCabinLetter(cabin):
#     match = cabinletter_matcher.search(cabin)
#     if match:
#         return ord(match.group())
#     else:
#         return 'U'
# 
# ### Find the number component of the cabin variable) 
# def getCabinNumber(cabin):
#     match = cabinnumber_matcher.search(cabin)
#     if match:
#         return float(match.group())
#     else:
#         return 0
#==================================================================================================================
    



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
input_df = input_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
test_df  = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 

print 'Building RandomForestClassifier with ' + str(len(input_df.columns)) \
      + ' columns: ' + str(list(input_df.columns.values))
    
print "Training examples: " + str(input_df.shape[0])

#==================================================================================================================
## Split training data into training and validation sets since we don't simple access to actual test set
# validation_rows = rnd.sample(input_df.index, int(input_df.shape[0] * 0.15))
# val_data = input_df.iloc[validation_rows].values
# train_data = input_df.drop(validation_rows).values
#==================================================================================================================
train_data = input_df.values
test_data = test_df.values

### Experiment with the number of estimators and number of examples used in the RandomForest
best_score = 0.0
best_score_e = 0
best_score_n = 0
train_scores = {}

e = 50              # number of estimators, test multiples of 50 up to 500
while e <= 500:
    #==============================================================================================================
    # n = 500
    # while n < train_data.shape[0]:
    #     print 'Training with ' + str(n) + " training examples and " + str(e) + ' estimators...'
    #     forest = RandomForestClassifier(n_estimators=e).fit( train_data[0:n,1::], train_data[0:n,0] )
    #     scores = cross_val_score(forest, train_data[0:n, 1::], train_data[0:n,0], cv=5)
    #     #print "cross_val_score() mean: " + str(scores.mean()) + " stddev: " + str(scores.std())
    #     if scores.mean() - scores.std() > best_score:
    #         best_score = scores.mean() - scores.std()
    #         best_score_e = e
    #         best_score_n = n
    #     n += 50
    #==============================================================================================================
         
    print 'Training with all training examples and ' + str(e) + ' estimators...'
    forest = RandomForestClassifier(n_estimators=e).fit( train_data[0::,1::], train_data[0::,0] )
    scores = cross_val_score(forest, train_data[0::, 1::], train_data[0::,0], cv=10)
    if scores.mean() - scores.std() > best_score:
            best_score = scores.mean() - scores.std()
            best_score_e = e
            best_score_n = train_data.shape[0]
    
    # set the score to optimize the lower bound as calculated by the cross validation
    train_scores[e] = scores.mean() - scores.std()
    
    e += 50

# Map the results of the estimators test
results_s = pd.Series(train_scores)
plt.scatter(results_s.keys(), results_s.values)
plt.xlabel("Number of Estimators")
plt.ylabel("Lower bound cross validation score")
plt.show(block=True)

print "Best score: " + str(best_score) + " found with estimators: " + str(best_score_e) \
        + ", examples: " + str(best_score_n)

# Run a final test with the best combination of examples and estimators as determined by cross validation
print 'Training with ' + str(best_score_n) + ' training examples and ' + str(best_score_e) + ' estimators...'
forest = RandomForestClassifier(n_estimators=best_score_e).fit( train_data[0:best_score_n,1::], \
                                                                train_data[0:best_score_n,0] )
scores = cross_val_score(forest, train_data[0:best_score_n, 1::], train_data[0:best_score_n,0], cv=5)
print "cross_val_score() mean: " + str(scores.mean()) + " stddev: " + str(scores.std())

# Predict the survival of the test set
print 'Predicting...'
output = forest.predict(test_data).astype(int)

# write results
predictions_file = open("data/results/randforest" + str(int(time.time())) + ".csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
