'''
Useful posts about ROC that helped me grok it:

http://fastml.com/what-you-wanted-to-know-about-auc/
http://scikit-learn.org/stable/auto_examples/plot_roc.html
http://en.wikipedia.org/wiki/Receiver_operating_characteristic
'''

import loaddata
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split


def generate_roc_curve(clf, X, y, survived_weight=1, plot=False):
    """
    Generates an ROC curve and calculates the AUC 
    """
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    weights = np.array([survived_weight if s == 1 else 1 for s in y_train])
    clf.fit(X_train, y_train, sample_weight=weights)
    
    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)

    print 'ROC AUC: %0.2f' % roc_auc

    if plot:
        # Plot of a ROC curve for a specific class
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()
    
    return roc_auc

if __name__ == "__main__":
    """
    Test method
    """
    print "Testing ROC Curve..."
    input_df, _ = loaddata.getDataSets(bins=True, scaled=True, binary=True)
    input_df.drop("PassengerId", axis=1, inplace=True)
    X = input_df.values[:,1::]
    y = input_df.values[:,0]
    forest = RandomForestClassifier(n_estimators=10000, n_jobs=-1)
    
    generate_roc_curve(forest, X, y)
