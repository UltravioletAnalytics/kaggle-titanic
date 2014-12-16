kaggle-titanic
==============

This is the python/scikit-learn code I wrote during my stab at the Kaggle titanic competition. There is code for several different algorithms, but the primary and highest performing one is the RandomForest implemented in randomforest2.py.

Requirements:
- python (a 2.x release at least 2.6)
- scikit-learn/NumPy/SciPy (http://scikit-learn.org/stable/install.html)
- pandas (http://pandas.pydata.org/pandas-docs/stable/install.html)
- matplotlib (http://matplotlib.org/faq/installing_faq.html)

Usage:<br/>
    > python randomforest2.py

Key files:
- loaddata.py: Contains all the feature engineering including options for generating different variable types, and performing PCA, clustering, and class balancing
- randomforest2.py: The code that executes the pipeline
- scorereport.py: Inspects and reports on the results of hyperparameter search
- learningcurve.py: Includes code to generate a learning curve
- roc_auc: Includes code to generate a ROC curve

Other files contain other algorithms that were used during experimentation and are in various stages of completeness. Only randomforest2 is 100% up to date
