"""RandomForest
"""
import loaddata
import scorereport
import learningcurve
import roc_auc
import numpy as np
import time
import csv
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def scoreForest(estimator, X, y):
    """
    Custom scoring function for hyperparameter optimization. In this case, we want to print out the oob score
    """
    score = estimator.oob_score_
    print "oob_score_:", score
    return score


if __name__ == '__main__':
    """
    Main script, this contains logic to execute the full pipeline to generate a RandomForest for the titanic data
    """
    ##############################################################################################################
    # Prepare data for pipeline
    #
    print "\nGenerating initial training/test sets"
    input_df, submit_df = loaddata.getDataSets(bins=True, scaled=True, binary=True)
    
    # Collect the test data's PassengerIds then drop it from the train and test sets
    submit_ids = submit_df['PassengerId']
            
    input_df.drop('PassengerId', axis=1, inplace=1) 
    submit_df.drop('PassengerId', axis=1, inplace=1)
    
    features_list = input_df.columns.values[1::] # Save for feature importance graph
    X = input_df.values[:, 1::]
    y = input_df.values[:, 0]
    
    # Set the weights to adjust for uneven class distributions (fewer passengers survived than died)
    survived_weight = .75
    y_weights = np.array([survived_weight if s == 1 else 1 for s in y])
    
    
    ##############################################################################################################
    # Reduce initial feature set with estimated feature importance
    #
    print "Rough fitting a RandomForest to determine feature importance..."
    forest = RandomForestClassifier(oob_score=True, n_estimators=10000)
    forest.fit(X, y, sample_weight=y_weights)
    feature_importance = forest.feature_importances_
    
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    #print "Feature importances:\n", feature_importance
    
    fi_threshold = 18
    
    important_idx = np.where(feature_importance > fi_threshold)[0]
    #print "Indices of most important features:\n", important_idx
    
    important_features = features_list[important_idx]
    print "\n", important_features.shape[0], "Important features(>", fi_threshold, "% of max importance)...\n"#, \
            #important_features
    
    sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
    
    # Plot feature importance
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]], align='center')
    plt.yticks(pos, important_features[sorted_idx[::-1]])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.draw()
    plt.show()
    
    # Remove non-important features from the feature set and submission sets
    X = X[:, important_idx][:, sorted_idx]
    #print "\nSorted (DESC) Useful X:\n", X
    
    submit_df = submit_df.iloc[:,important_idx].iloc[:,sorted_idx]
    print '\nTraining with', X.shape[1], "features:\n", submit_df.columns.values
    #print input_df.iloc[:,1::].iloc[:,important_idx].iloc[:,sorted_idx].head(10)
    
        
    ##############################################################################################################
    # Hyperparameter Optimization - uncomment one of the algorithms below to implement and update the "params"
    # assignment in order to run optimization (which can take a while!)
    #
    sqrtfeat = int(np.sqrt(X.shape[1]))
    #print "sqrtfeat:", sqrtfeat
    minsampsplit = int(X.shape[0]*0.015)
    #print "minsampsplit:", minsampsplit
    
    
    # specify model parameters and distributions to sample from
    #==============================================================================================================
    # params_test = { "n_estimators"      : np.rint(np.linspace(X.shape[0]*2, X.shape[0]*4, 4)).astype(int),
    #                 "max_features"      : np.rint(np.linspace(sqrtfeat, sqrtfeat*2, 3)).astype(int),
    #                 "min_samples_split" : np.rint(np.linspace(2, X.shape[0]/50, 4)).astype(int) }
    #==============================================================================================================

    params_test = { "n_estimators"      : [5000, 10000],
                    "max_features"      : np.rint(np.linspace(sqrtfeat, sqrtfeat+2, 3)).astype(int),
                    "min_samples_split" : np.rint(np.linspace(X.shape[0]*.01, X.shape[0]*.05, 3)).astype(int) }
    
    params_score = { "n_estimators"      : 10000,
                     "max_features"      : sqrtfeat,
                     "min_samples_split" : minsampsplit }
    
    #==============================================================================================================
    # print "Hyperparameter optimization using RandomizedSearchCV..."
    # rand_search = RandomizedSearchCV(forest, params, n_jobs=-1, n_iter=20)
    # rand_search.fit(X, y)
    # best_params = report(rand_search.grid_scores_)
    #==============================================================================================================
    
    #==============================================================================================================
    # print "Hyperparameter optimization using GridSearchCV..."
    # grid_search = GridSearchCV(forest, params_test, scoring=scoreForest, n_jobs=-1, cv=10,
    #                            fit_params={ 'sample_weight': y_weights})
    # grid_search.fit(X, y)
    # best_params = scorereport.report(grid_search.grid_scores_)
    #==============================================================================================================

    # Use parameters from either the hyperparameter optimization, or manually selected parameters...
    params = params_score
    #params = best_params
    
    
    
    #############################################################################################################
    # Model generation/validation
    #
    print "Generating RandomForestClassifier model with parameters: ", params
    forest = RandomForestClassifier(n_jobs=-1, oob_score=True, **params)
    
    
    print "\nCalculating Learning Curve..."
    title = "RandomForestClassifier with hyperparams: ", params
    midpoint, diff = \
         learningcurve.plot_learning_curve(forest, title, X, y, (0.6, 1.01), cv=8, n_jobs=-1, plot=True)
    #print "Midpoint:", midpoint
    #print "Diff:", diff
    
        
    print "\nGenerating ROC curve 5 times to get mean AUC with class weights..."
    aucs = []
    for i in range(5):
        aucs.append(roc_auc.generate_roc_curve(forest, X, y, survived_weight))
    auc_mean = ("%.3f"%(np.mean(aucs))).lstrip('0')
    auc_std = ("%.3f"%(np.std(aucs))).lstrip('0')
    auc_lower = ("%.3f"%(np.mean(aucs)-np.std(aucs))).lstrip('0')
    print "ROC - Area under curve:", auc_mean, "and stddev:", auc_std
    
    
    print "\nFitting model 5 times to get mean OOB score using full training data with class weights..."
    test_scores = []
    # Using the optimal parameters, predict the survival of the labeled test set 10 times
    for i in range(5):
        forest.fit(X, y, sample_weight=y_weights)
        print "OOB:", forest.oob_score_
        test_scores.append(forest.oob_score_)
    oob = ("%.3f"%(np.mean(test_scores))).lstrip('0')
    oob_std = ("%.3f"%(np.std(test_scores))).lstrip('0')
    oob_lower = ("%.3f"%(np.mean(test_scores) - np.std(test_scores))).lstrip('0')
    print "OOB Mean:", oob, "and stddev:", oob_std
    print "Est. correctly identified test examples:", np.mean(test_scores) * X.shape[0]
    
    
    ###########################################################################################################
    # Final prediction and save results
    #
    print "\nSubmitting predicted labels for", submit_df.shape[0], "records with class weights..."
    submission = np.asarray(zip(submit_ids, forest.predict(submit_df))).astype(int)
    
    print "Survived weight:", survived_weight
    srv_pct = "%.3f"%(submission[:,1].mean())
    print "Died/Survived: ", "%.3f"%(1-submission[:,1].mean()) , "/", srv_pct
    
    # sort to ensure the passenger IDs are in the correct sequence
    output = submission[submission[:,0].argsort()]
    
    # write results to a file
    name = "rfc" + str(int(time.time())) + "_AUC-" + auc_lower + "_OOB-" + oob_lower + "_SRV(" +\
            str(survived_weight) + ")-" + srv_pct + ".csv"
    print "Generating results file:", name
    predictions_file = open("data/results/" + name, "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(output)
    
    print 'Done.'
