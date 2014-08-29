import re
import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing



# Globals
#############################
cabinletter_matcher = re.compile("([a-zA-Z]+)")
cabinnumber_matcher = re.compile("([0-9]+)")



# Functions
#############################

### Cabin numbers, when present, contain a single (or space-delimited list) cabin number that is composed of
### a letter and number with no space or other character between.
def processCabin(df):     
    
    # Replace missing values with "U0"
    df['Cabin'][df.Cabin.isnull()] = 'U0'
    
    #==============================================================================================================
    # # create features for the alphabetical part of the cabin number
    # df['CabinLetter'] = df['Cabin'].map( lambda x : getCabinLetter(x))
    # # create binary features for all the distinct cabin letters
    # if keep_binary:
    #     for letter in df.CabinLetter.unique():
    #         df['CabinLetter_' + str(letter)] = np.where(df['CabinLetter'] == letter, 1, 0)
    #     
    # # create feature for the numerical part of the cabin number
    # df['CabinNumber'] = df['Cabin'].map( lambda x : getCabinNumber(x))
    # # scale the number to process as a continuous feature
    # if keep_scaled:
    #     scaler = preprocessing.StandardScaler()
    #     df['CabinNumber_scaled'] = scaler.fit_transform(df['CabinNumber'])
    #==============================================================================================================

    if not keep_raw:
        df.drop(['Cabin'], axis=1, inplace=True)
        #df.drop(['Cabin', 'CabinLetter', 'CabinNumber'], axis=1, inplace=True)

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
        return match.group()
    else:
        return 0


### Build 5 bins for Ages to create binary features
###   param df - contains the entire Dataframe with all data from train and test
def processAge(df):
    setMissingAges(df)
    
    # center the mean and scale to unit variance
    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Age_scaled'] = df['Age']
        scaler.fit_transform(df['Age_scaled'])
    
    # have a feature for children
    #==============================================================================================================
    # df['isChild'] = df[np.where(df.Age < 18, 1, 0)]
    #==============================================================================================================
    
    # bin into quintiles and create binary features
    df['Age_bin'] = pd.qcut(df['Age'], 4)
    if keep_binary:
        for ac in df.Age_bin.unique():
            df['Age_bin_' + str(ac)] = np.where(df['Age_bin'] == ac, 1, 0)
    if not keep_bins:
        df.drop('Age_bin', axis=1, inplace=True)
    
    if not keep_raw:
        df.drop('Age', axis=1, inplace=True)
    


### Populate missing ages by using median values per gender and class
###   param df - contains the entire Dataframe with all data from train and test
def setMissingAges(df):    
    # calculate median ages by gender and class
    grouped = df[['Sex', 'Pclass', 'Age']].dropna().groupby(['Sex', 'Pclass']).mean().astype(int)
    
    # get all rows with missing Age values
    noage = df[['Sex', 'Pclass', 'Age']].loc[ (df.Age.isnull()) ]
    
    # set all missing ages by looking up the median calculated earlier
    df.loc[ (df.Age.isnull()), 'Age'] = noage.apply(lambda x: grouped.loc[x['Sex']].loc[x['Pclass']]['Age'], axis=1)


### Build 5 bins for ticket prices to create binary features
###   param df - contains the entire Dataframe with all data from train and test
def processFare(df):        
    # replace missing values as the median fare. Currently the datasets only contain one missing Fare value
    df['Fare'][ np.isnan(df['Fare']) ] = df['Fare'].median()
    
    # center and scale the fare to use as a continuous variable
    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Fare_scaled'] = df['Fare']
        scaler.fit_transform(df['Fare_scaled'])
    
    # bin into quintiles for binary features
    df['Fare_bin'] = pd.qcut(df['Fare'], 4)
    if keep_binary:
        for farebin in df.Fare_bin.unique():
            df['Fare_bin_' + str(farebin)] = np.where(df['Fare_bin'] == farebin, 1, 0)
    if not keep_bins:
        df.drop('Fare_bin', axis=1, inplace=True)
    
    if not keep_raw:
        df.drop('Fare', axis=1, inplace=True)


# Build binary features from 3-valued categorical feature
###   param df - contains the entire Dataframe with all data from train and test
def processEmbarked(df):
    # Replace missing values with most common port, and create binary features
    df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values
    
    # Lets turn this into a number so it conforms to decision tree feature requirements
    df['Embarked'] = df['Embarked'].apply(lambda x: ord(x))
        
    #==============================================================================================================
    # # Create binary features for each port
    # if keep_binary:
    #     for port in df.Embarked.unique():
    #         df['Embarked_' + str(port)] = np.where(df['Embarked'] == port, 1, 0)
    #==============================================================================================================

    if not keep_raw:
        df.drop('Embarked', axis=1, inplace=True)

def processPClass(df):
    # Replace missing values with mode
    df.Pclass[ df.Pclass.isnull() ] = df.Pclass.dropna().mode().values
    
    #==============================================================================================================
    # # create binary features
    # if keep_binary:
    #     for pc in df.Pclass.unique():
    #         df['Pclass_' + str(pc)] = np.where(df['Pclass'] == pc, 1, 0)
    #==============================================================================================================

    if not keep_raw:
        df.drop('Pclass', axis=1, inplace=True)

def processFamily(df):
    #==============================================================================================================
    # # First process scaling
    # if keep_scaled:
    #     scaler = preprocessing.StandardScaler()
    #     df['SibSp_scaled'] = scaler.fit_transform(df['SibSp'])
    #     df['Parch_scaled'] = scaler.fit_transform(df['Parch'])
    # 
    # # Then build binary features
    # if keep_binary:
    #     for s in df.SibSp.unique():
    #         df['SibSp_bin_' + str(s)] = np.where(df['SibSp'] == s, 1, 0)
    #     for p in df.Parch.unique():
    #         df['Parch_bin_' + str(p)] = np.where(df['Parch'] == p, 1, 0)
    # 
    #==============================================================================================================
    
    # perhaps the number of siblings/spouses and parents/children aren't as important as the total family size
    df['familySize'] = df.SibSp + df.Parch
    
    if not keep_raw:
        df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

def processSex(df):
    df['Gender'] = np.where(df['Sex'] == 'male', 1, 0)
    
    if not keep_raw:
        df.drop('Sex', axis=1, inplace=True)

def processUnused(df):
    if not keep_raw:
        df.drop(['Name', 'Ticket'], axis=1, inplace=True)

### Main script for dataset generation.
### returns two DataFrame objects
def getDataSets(binary=True, bins=True, scaled=True, raw=True):
    global keep_binary, keep_bins, keep_scaled, keep_raw
    keep_binary = binary
    keep_bins = bins
    keep_scaled = scaled
    keep_raw = raw
    
    # read in the training and testing data into Pandas.DataFrame objects
    input_df = pd.read_csv('data/raw/train.csv', header=0)
    test_df  = pd.read_csv('data/raw/test.csv',  header=0)
    
    full_df = pd.concat([input_df, test_df])
    
    processAge(full_df)
    processFare(full_df)    
    processEmbarked(full_df)    
    processFamily(full_df)
    processCabin(full_df)
    processSex(full_df)
    processPClass(full_df)
    processUnused(full_df)
    
    # Move the survived column back to the first position
    columns_list = list(full_df.columns.values)
    columns_list.remove('Survived')
    new_col_list = list(['Survived'])
    new_col_list.extend(columns_list)
    full_df = full_df.reindex(columns=new_col_list)
    
    input_df = full_df[:input_df.shape[0]] 
    test_df  = full_df[input_df.shape[0]:]
    
    return input_df, test_df

train, test = getDataSets(raw=False, bins=False, scaled=False)
