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
import random as rd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Functions
#############################

### Cabin numbers, when present, contain a single (or space-delimited list) cabin number that is composed of
### a letter and number with no space or other character between. This is a sparse variable: < 30% is populated
def processCabin():     
    global df
    # Replace missing values with "U0"
    df['Cabin'][df.Cabin.isnull()] = 'U0'
    
    # create features for the alphabetical part of the cabin number
    df['CabinLetter'] = df['Cabin'].map( lambda x : getCabinLetter(x))
    df['CabinLetter'][df.CabinLetter == 84] = 85 # Cabin 84 only occurs once in combined set to change to 85 (Unknown)
    df['CabinLetter'][df.CabinLetter == 71] = 70 # Cabin 71 only occurs 7 times, combine with cabin 70
    
    
    # create binary features for each cabin letters
    if keep_binary:
        cletters = pd.get_dummies(df['CabinLetter']).rename(columns=lambda x: 'CabinLetter_' + str(x))
        df = pd.concat([df, cletters], axis=1)
    
    # create feature for the numerical part of the cabin number
    df['CabinNumber'] = df['Cabin'].map( lambda x : getCabinNumber(x)).astype(int)
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
    

### Build 5 bins for ticket prices to create binary features
###   param df - contains the entire Dataframe with all data from train and test
def processFare():
    global df
            
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
        df = pd.concat([df, pd.get_dummies(df['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))], axis=1)
    
    if keep_bins:
        df['Fare_bin_id'] = pd.factorize(df['Fare_bin'])[0]
    
    if not keep_strings:
        df.drop('Fare_bin', axis=1, inplace=True)
    

# Build binary features from 3-valued categorical feature
###   param df - contains the entire Dataframe with all data from train and test
def processEmbarked():
    global df
    
    # Replace missing values with most common port, and create binary features
    df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values
    
    # Lets turn this into a number so it conforms to decision tree feature requirements
    df['Embarked'] = df['Embarked'].apply(lambda x: ord(x))
        
    # Create binary features for each port
    if keep_binary:
        df = pd.concat([df, pd.get_dummies(df['Embarked']).rename(columns=lambda x: 'Embarked_' + str(x))], axis=1)
   

def processPClass():
    global df
    
    # Replace missing values with mode
    df.Pclass[ df.Pclass.isnull() ] = df.Pclass.dropna().mode().values
    
    # create binary features
    if keep_binary:
        df = pd.concat([df, pd.get_dummies(df['Pclass']).rename(columns=lambda x: 'Pclass_' + str(x))], axis=1)


def processFamily():
    global df
    
    # perhaps the number of siblings/spouses and parents/children aren't as important as the total family size
    df['FamilySize'] = df.SibSp + df.Parch
        
    # First process scaling
    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['SibSp_scaled'] = scaler.fit_transform(df['SibSp'])
        df['Parch_scaled'] = scaler.fit_transform(df['Parch'])
        df['FamilySize_scaled'] = scaler.fit_transform(df['FamilySize'])
     
    # Then build binary features
    if keep_binary:
        sibsps = pd.get_dummies(df['SibSp']).rename(columns=lambda x: 'SibSp_' + str(x))
        parchs = pd.get_dummies(df['Parch']).rename(columns=lambda x: 'Parch_' + str(x))
        fsizes = pd.get_dummies(df['FamilySize']).rename(columns=lambda x: 'FamilySize_' + str(x))
        df = pd.concat([df, sibsps, parchs, fsizes], axis=1)
    

def processSex():
    global df
    df['Gender'] = np.where(df['Sex'] == 'male', 1, 0)
    

def processName():
    global df
    # how many different names do they have? 
    df['Names'] = df['Name'].map(lambda x: len(re.split(' ', x)))
    
    # what is each person's title? Group low-occuring related titles together
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
        df = pd.concat([df, pd.get_dummies(df['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis=1)
    
    if not keep_strings:
        df['TitleId'] = pd.factorize(df['Title'])[0]
    

def getTitle(name):
    match = re.compile(", (.*?)\.").findall(name)
    return np.nan if not match else match[0]


def processUnused():
    global df
    if not keep_raw:
        df.drop(['Ticket'], axis=1, inplace=True)


def processComposite():
    global df
    #farePerPerson
    df['FarePerPerson'] = df['Fare'] / (df['FamilySize'] + 1)
    # scale the number to process as a continuous feature
    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['FarePerPerson_scaled'] = scaler.fit_transform(df['FarePerPerson'])


### Build 5 bins for Ages to create binary features
###   param df - contains the entire Dataframe with all data from train and test
def processAge():
    global df
    setMissingAges()
    
    # center the mean and scale to unit variance
    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Age_scaled'] = df['Age']
        scaler.fit_transform(df['Age_scaled'])
    
    # have a feature for children
    df['isChild'] = np.where(df.Age < 13, 1, 0)
    
    # bin into quartiles and create binary features
    df['Age_bin'] = pd.qcut(df['Age'], 4)
    if keep_binary:
        df = pd.concat([df, pd.get_dummies(df['Age_bin']).rename(columns=lambda x: 'Age_' + str(x))], axis=1)
    
    if keep_bins:
        df['Age_bin_id'] = pd.factorize(df['Age_bin'])[0]
    
    if not keep_strings:
        df.drop('Age_bin', axis=1, inplace=True)
    

### Populate missing ages  using RandomForestClassifier
def setMissingAges():
    global df
    
    age_df = df[['Age','Embarked','Fare','FamilySize','FarePerPerson','TitleId','Pclass','Names','CabinLetter']]
    X = age_df.loc[ (df.Age.notnull()) ].values[:, 1::]
    y = age_df.loc[ (df.Age.notnull()) ].values[:, 0]
    
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)
    
    #==============================================================================================================
    # # Plot feature importance
    # feature_importance = rtr.feature_importances_
    # # make importances relative to max importance
    # feature_importance = 100.0 * (feature_importance / feature_importance.max())
    # sorted_idx = np.argsort(feature_importance)
    # pos = np.arange(sorted_idx.shape[0]) + .5
    # plt.subplot(1, 2, 2)
    # plt.barh(pos, feature_importance[sorted_idx], align='center')
    # plt.yticks(pos, age_df.columns.values[1::][sorted_idx])
    # plt.xlabel('Relative Importance')
    # plt.title('Variable Importance')
    # plt.draw()
    # plt.show()
    #==============================================================================================================
    
    predictedAges = rtr.predict(age_df.loc[ (df.Age.isnull()) ].values[:, 1::])
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 


# Keep the raw list until the very end even if raw values are not retained so that interaction
# parameters can be created
def processDrops():
    global df
    rawDropList = ['Name', 'Names', 'Title', 'Sex', 'SibSp', 'Parch', 'FamilySize', 'Pclass', 'Embarked', \
                   'Cabin', 'CabinLetter', 'CabinNumber', 'Age', 'Fare', 'FarePerPerson']
    stringsDropList = ['Title', 'Name', 'Cabin', 'Ticket', 'Sex']
    if not keep_raw:
        df.drop(rawDropList, axis=1, inplace=True)
    elif not keep_strings:
        df.drop(stringsDropList, axis=1, inplace=True)

### Performs all feature engineering tasks including populating missing values, generating binary categorical
### features, scaling, and other transformations
def getDataSets(binary=False, bins=False, scaled=False, raw=True, pca=False, balanced=False, strings=False):
    global keep_binary, keep_bins, keep_scaled, keep_raw, keep_strings, df
    keep_binary = binary
    keep_bins = bins
    keep_scaled = scaled
    keep_raw = raw
    keep_strings = strings
    
    # read in the training and testing data into Pandas.DataFrame objects
    input_df = pd.read_csv('data/raw/train.csv', header=0)
    submit_df  = pd.read_csv('data/raw/test.csv',  header=0)
    
    df = pd.concat([input_df, submit_df])
    
    processName()
    processFare()    
    processEmbarked()    
    processFamily()
    processCabin()
    processSex()
    processPClass()
    processUnused()
    processComposite()
    processAge()
    processDrops()
    
    # Move the survived column back to the first position
    columns_list = list(df.columns.values)
    columns_list.remove('Survived')
    new_col_list = list(['Survived'])
    new_col_list.extend(columns_list)
    df = df.reindex(columns=new_col_list)
    
    # Check out the correlation
    df_sper = df.corr(method='spearman')
    print df_sper > .9
    sys.exit()
    
    input_df = df[:input_df.shape[0]] 
    submit_df  = df[input_df.shape[0]:]
    
    if pca:
        print "reducing and clustering now..."
        input_df, submit_df = reduceAndCluster(input_df, submit_df)
    else:
        # drop the empty 'Survived' column for the test set that was created during set concatentation
        submit_df.drop('Survived', axis=1, inplace=1)
    
    print input_df.columns.size, "features generated set before Reduction and Clustering: ", input_df.columns.values
    
    if balanced:
        # Undersample training examples of passengers who did not survive
        print 'Perished data shape:', input_df[input_df.Survived==0].shape
        print 'Survived data shape:', input_df[input_df.Survived==1].shape
        perished_sample = rd.sample(input_df[input_df.Survived==0].index, input_df[input_df.Survived==1].shape[0])
        input_df = pd.concat([input_df.ix[perished_sample], input_df[input_df.Survived==1]])
        input_df.sort(inplace=True)
        print 'New even class training shape:', input_df.shape
    
    return input_df, submit_df


### Takes the train and test data frames and performs dimensionality reduction and clustering
def reduceAndCluster(input_df, submit_df, clusters=2):
    # join the full data together
    df = pd.concat([input_df, submit_df])
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    df = df.reindex_axis(input_df.columns, axis=1)
    
    # Split into feature and label arrays
    X = df.values[:, 1::]
    y = df.values[:, 0]
    
    # Series of labels
    survivedSeries = pd.Series(df['Survived'], name='Survived')
    
    # Run PCA for dimensionality reduction. Look for smallest number of parameters that explain 99% of variance
    for c in range(2,df.columns.size-1):
        pca = PCA(n_components=c, whiten=True).fit(X,y)
        if pca.explained_variance_ratio_.sum() > .99:
            print c, " components describe ", pca.explained_variance_ratio_.sum(), "% of the variance"
            break
    
    # transform the initial features into fewer parameters that provide nearly the same variance explanatory power
    X_pca = pca.transform(X)
    pcaDataFrame = pd.DataFrame(X_pca)
    
    # use basic clustering to group similar examples and save the cluster ID for each example in train and test
    kmeans = KMeans(n_clusters=clusters, random_state=np.random.RandomState(4), init='random')
     
    #==============================================================================================================
    # # Perform clustering on labeled AND unlabeled data
    # clusterIds = kmeans.fit_predict(X_pca)
    #==============================================================================================================
    
    # Perform clustering on labeled data and then predict clusters for unlabeled data
    trainClusterIds = kmeans.fit_predict(X_pca[:input_df.shape[0]])
    print "clusterIds shape for training data: ", trainClusterIds.shape
    #print "trainClusterIds: ", trainClusterIds
     
    testClusterIds = kmeans.predict(X_pca[input_df.shape[0]:])
    print "clusterIds shape for test data: ", testClusterIds.shape
    #print "testClusterIds: ", testClusterIds
     
    clusterIds = np.concatenate([trainClusterIds, testClusterIds])
    print "all clusterIds shape: ", clusterIds.shape
    #print "clusterIds: ", clusterIds
    
    
    # construct the new DataFrame comprised of "Survived", "ClusterID", and the PCA features
    clusterIdSeries = pd.Series(clusterIds, name='ClusterId')
    df = pd.concat([survivedSeries, clusterIdSeries, pcaDataFrame], axis=1)
    
    # split into separate input and test sets again
    input_df = df[:input_df.shape[0]]
    submit_df = df[input_df.shape[0]:]
    submit_df.reset_index(inplace=True)
    submit_df.drop('index', axis=1, inplace=True)
    submit_df.drop('Survived', axis=1, inplace=1)
    
    print "Labeled survived counts :\n", pd.value_counts(input_df['Survived'])/input_df.shape[0]
    print "Labeled cluster counts  :\n", pd.value_counts(input_df['ClusterId'])/input_df.shape[0]
    print "Unlabeled cluster counts:\n", pd.value_counts(submit_df['ClusterId'])/submit_df.shape[0]
    
    return input_df, submit_df


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    train, test = getDataSets(bins=True, binary=True)
    drop_list = ['PassengerId']
    train.drop(drop_list, axis=1, inplace=1) 
    test.drop(drop_list, axis=1, inplace=1)
    
    print train.drop('Survived', axis=1).columns.values
    pd.set_option('display.max_columns', None)
    print train.drop('Survived', axis=1).head()
    
    train, test = reduceAndCluster(train, test)
    
    print train.columns.values
