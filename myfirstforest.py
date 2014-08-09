""" Writing my first randomforest code.
Author : AstroDave
Date : 23rd September 2012
Revised: 15 April 2014
please see packages.python.org/milk/randomforests.html for more

""" 
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier


# Globals
############################3
ports_dict = {}               # Holds the possible values of 'Embarked' variable





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
input_df = input_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], \
                         axis=1) 
test_df  = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], \
                        axis=1) 

print 'Building RandomForestClassifier with ' + str(len(input_df.columns)) \
      + ' columns: ' + str(list(input_df.columns.values))
    
# convert DataFrames back to numpy arrays
train_data = input_df.values
test_data = test_df.values

print 'Training...'
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )
forest = forest.score( train_data[0::,1::], train_data[0::,0] )
print 'oob_score_: ' + str(forest.oob_score_)

print 'Predicting...'
output = forest.predict(test_data).astype(int)

# write results
predictions_file = open("data/results/myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
