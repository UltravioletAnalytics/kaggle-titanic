'''
Handles all the data preparation including: feature engineering, dimensionality reduction, and clustering

Inspiration for the feature engineering had several sources:

http://trevorstephens.com/post/73461351896/titanic-getting-started-with-r-part-4-feature
http://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/
http://www.sgzhaohang.com/blog/tag/kaggle/
'''
import re
import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Functions
#############################

### Cabin numbers, when present, contain a single (or space-delimited list) cabin number that is composed of
### a letter and number with no space or other character between. This is a sparse variable: < 30% is populated
def processCabin(df):     
    
    # Replace missing values with "U0"
    df['Cabin'][df.Cabin.isnull()] = 'U0'
    
    # create features for the alphabetical part of the cabin number
    df['CabinLetter'] = df['Cabin'].map( lambda x : getCabinLetter(x))
    df['CabinLetter'][df.CabinLetter == 84] = 85 # Cabin 84 only occurs once in combined set to change to 85 (Unknown)
    df['CabinLetter'][df.CabinLetter == 71] = 70 # Cabin 71 only occurs 7 times, combine with cabin 70
    
    
    # create binary features for each cabin letters
    if keep_binary:
        for letter in df.CabinLetter.unique():
            df['CabinLetter_' + str(letter)] = np.where(df['CabinLetter'] == letter, 1, 0)
    
    
    # create feature for the numerical part of the cabin number
    df['CabinNumber'] = df['Cabin'].map( lambda x : getCabinNumber(x))
    # scale the number to process as a continuous feature
    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['CabinNumber_scaled'] = scaler.fit_transform(df['CabinNumber'])

    

### Find the letter component of the cabin variable) 
def getCabinLetter(cabin):
    match = re.compile("([a-zA-Z]+)").search(cabin)
    if match:
        return ord(match.group())
    else:
        return 'U'


### Find the number component of the cabin variable) 
def getCabinNumber(cabin):
    match = re.compile("([0-9]+)").search(cabin)
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
    
    # bin into quartiles and create binary features
    df['Age_bin'] = pd.qcut(df['Age'], 4)
    if keep_binary:
        for ac in df.Age_bin.unique():
            df['Age_' + str(ac)] = np.where(df['Age_bin'] == ac, 1, 0)
    
    if not keep_bins:
        df.drop('Age_bin', axis=1, inplace=True)
    

### Populate missing ages by using median values per title and class
###   param df - contains the entire Dataframe with all data from train and test
def setMissingAges(df):    
    # calculate median ages by gender and class
    grouped = df[['Title', 'Pclass', 'Age']].dropna().groupby(['Title', 'Pclass']).mean().astype(int)
    
    # get all rows with missing Age values
    noage = df[['Title', 'Pclass', 'Age']].loc[ (df.Age.isnull()) ]
    
    # set all missing ages by looking up the median calculated earlier
    df.loc[ (df.Age.isnull()), 'Age'] = noage.apply(lambda x: grouped.loc[x['Title']].loc[x['Pclass']]['Age'], axis=1)
    

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
    
    #==============================================================================================================
    # # bin into quintiles for binary features
    # df['Fare_bin'] = pd.qcut(df['Fare'], 4)
    # if keep_binary:
    #     for farebin in df.Fare_bin.unique():
    #         df['Fare_bin_' + str(farebin)] = np.where(df['Fare_bin'] == farebin, 1, 0)
    # if not keep_bins:
    #     df.drop('Fare_bin', axis=1, inplace=True)
    #==============================================================================================================
    

# Build binary features from 3-valued categorical feature
###   param df - contains the entire Dataframe with all data from train and test
def processEmbarked(df):
    # Replace missing values with most common port, and create binary features
    df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values
    
    # Lets turn this into a number so it conforms to decision tree feature requirements
    df['Embarked'] = df['Embarked'].apply(lambda x: ord(x))
        
    # Create binary features for each port
    if keep_binary:
        for port in df.Embarked.unique():
            df['Embarked_' + str(port)] = np.where(df['Embarked'] == port, 1, 0)
   

def processPClass(df):
    # Replace missing values with mode
    df.Pclass[ df.Pclass.isnull() ] = df.Pclass.dropna().mode().values
    
    # create binary features
    if keep_binary:
        for pc in df.Pclass.unique():
            df['Pclass_' + str(pc)] = np.where(df['Pclass'] == pc, 1, 0)


def processFamily(df):
    # perhaps the number of siblings/spouses and parents/children aren't as important as the total family size
    df['FamilySize'] = df.SibSp + df.Parch
        
    # First process scaling
    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['SibSp_scaled'] = scaler.fit_transform(df['SibSp'])
        df['Parch_scaled'] = scaler.fit_transform(df['Parch'])
        df['FamilySize_scaled'] = scaler.fit_transform(df['FamilySize'])
     
    #==============================================================================================================
    # # Then build binary features
    # if keep_binary:
    #     for s in df.SibSp.unique():
    #         df['SibSp_bin_' + str(s)] = np.where(df['SibSp'] == s, 1, 0)
    #     for p in df.Parch.unique():
    #         df['Parch_bin_' + str(p)] = np.where(df['Parch'] == p, 1, 0)
    #==============================================================================================================
    

def processSex(df):
    df['Gender'] = np.where(df['Sex'] == 'male', 1, 0)
    

def processName(df):
    # how many different names do they have? 
    df['Names'] = df['Name'].map(lambda x: len(re.split(' ', x)))
    
    # what is each person's title? Group low-occuring titles together
    df['Title'] = df['Name'].map(lambda x: getTitle(x))
    df['Title'][df.Title == 'Jonkheer'] = 'Master'
    df['Title'][df.Title.isin(['Ms','Mlle'])] = 'Miss'
    df['Title'][df.Title == 'Mme'] = 'Mrs'
    df['Title'][df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
    df['Title'][df.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
    
    # process scaling
    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Names_scaled'] = scaler.fit_transform(df['Names'])
    
    # Build binary features
    if keep_binary:
        for t in df.Title.unique():
            df['Title_' + str(t)] = np.where(df['Title'] == t, 1, 0)
    

def getTitle(name):
    match = re.compile(", (.*?)\.").findall(name)
    return np.nan if not match else match[0]


def processUnused(df):
    if not keep_raw:
        df.drop(['Ticket'], axis=1, inplace=True)


def processComposite(df):
    #farePerPerson
    df['FarePerPerson'] = df['Fare'] / (df['FamilySize'] + 1)
    # scale the number to process as a continuous feature
    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['FarePerPerson_scaled'] = scaler.fit_transform(df['FarePerPerson'])


# Keep the raw list until the very end even if raw values are not retained so that interaction
# parameters can be created
def processDrops(df):
    rawDropList = ['Name', 'Names', 'Title', 'Sex', 'SibSp', 'Parch', 'FamilySize', 'Pclass', 'Embarked', \
                   'Cabin', 'CabinLetter', 'CabinNumber', 'Age', 'Fare', 'FarePerPerson']
    if not keep_raw:
        df.drop(rawDropList, axis=1, inplace=True)

### Performs all feature engineering tasks including populating missing values, generating binary categorical
### features, scaling, and other transformations
def getDataSets(binary=True, bins=True, scaled=True, raw=True, pca=False):
    global keep_binary, keep_bins, keep_scaled, keep_raw
    keep_binary = binary
    keep_bins = bins
    keep_scaled = scaled
    keep_raw = raw
    
    # read in the training and testing data into Pandas.DataFrame objects
    input_df = pd.read_csv('data/raw/train.csv', header=0)
    test_df  = pd.read_csv('data/raw/test.csv',  header=0)
    
    full_df = pd.concat([input_df, test_df])
    
    processName(full_df)
    processAge(full_df)
    processFare(full_df)    
    processEmbarked(full_df)    
    processFamily(full_df)
    processCabin(full_df)
    processSex(full_df)
    processPClass(full_df)
    processUnused(full_df)
    processComposite(full_df)
    processDrops(full_df)
    
    # Move the survived column back to the first position
    columns_list = list(full_df.columns.values)
    columns_list.remove('Survived')
    new_col_list = list(['Survived'])
    new_col_list.extend(columns_list)
    full_df = full_df.reindex(columns=new_col_list)
    
    input_df = full_df[:input_df.shape[0]] 
    test_df  = full_df[input_df.shape[0]:]
    
    if pca:
        print "reducing and clustering now..."
        input_df, test_df = reduceAndCluster(input_df, test_df)
    else:
        # drop the empty 'Survived' column for the test set that was created during set concatentation
        test_df.drop('Survived', axis=1, inplace=1)
    
    
    return input_df, test_df


### Takes the train and test data frames and performs dimensionality reduction and clustering
def reduceAndCluster(input_df, test_df):
    # join the full data together
    full_df = pd.concat([input_df, test_df])
    full_df.reset_index(inplace=True)
    full_df.drop('index', axis=1, inplace=True)
    full_df = full_df.reindex_axis(input_df.columns, axis=1)
    
    # Split into feature and label arrays
    X = full_df.values[:, 1::]
    y = full_df.values[:, 0]
    
    # Series of labels
    survivedSeries = pd.Series(full_df['Survived'], name='Survived')
    
    # Run PCA for dimensionality reduction. Look for smallest number of parameters that explain 99% of variance
    for c in range(2,full_df.columns.size-1):
        pca = PCA(n_components=c, whiten=True).fit(X,y)
        print c, " components describe ", pca.explained_variance_ratio_.sum(), "% of the variance"
        if pca.explained_variance_ratio_.sum() > .99:
            break
    
    # transform the initial features into fewer parameters that provide nearly the same variance explanatory power
    X_pca = pca.transform(X)
    pcaDataFrame = pd.DataFrame(X_pca)
    
    # use basic clustering to group similar examples and save the cluster ID for each example in train and test
    # *** Should I be clustering all together, or fit the train set alone and predict the test set? 
    kmeans = KMeans(n_clusters=3, random_state=np.random.RandomState(17))
    kmeans.fit(X_pca)
    # ***kmeans.predict(<<just the training examples>>)
    clusterIdSeries = pd.Series(kmeans.labels_, name='ClusterId')
        
    # construct the new DataFrame comprised of "Survived", "ClusterID", and the PCA features
    full_df = pd.concat([survivedSeries, clusterIdSeries, pcaDataFrame], axis=1)
        
    # split into separate input and test sets again
    input_df = full_df[:input_df.shape[0]]
    test_df = full_df[input_df.shape[0]:]
    test_df.reset_index(inplace=True)
    test_df.drop('index', axis=1, inplace=True)
    test_df.drop('Survived', axis=1, inplace=1)
    
    #print pd.value_counts(input_df['ClusterId'])/input_df.shape[0]
    #print pd.value_counts(test_df['ClusterId'])/test_df.shape[0]
    
    return input_df, test_df

if __name__ == '__main__':
    train, test = getDataSets(raw=False, binary=True, bins=False, scaled=True)
    drop_list = ['PassengerId']
    train.drop(drop_list, axis=1, inplace=1) 
    test.drop(drop_list, axis=1, inplace=1) 
    train, test = reduceAndCluster(train, test)
