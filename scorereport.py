import numpy as np
from operator import itemgetter


def report(grid_scores, n_top=10):
    """
    Output a simple report of the top parameter sets from hyperparameter optimization
    
    adapted from http://scikit-learn.org/stable/auto_examples/randomized_search.html
    
    Parameters
    ----------
    grid_scores : array-like, shape (# of combinations tested)
        The scores of all parameter sets tested in the grid/randomized search

    n_top : int
        The top N number of models we want to report results for. Defaults to top 10.
    """
    params = None
    #top_scores = sorted(grid_scores, cmp=compare_scores, reverse=True)[:n_top] # custom sorting
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Parameters with rank: {0}".format(i + 1))
        #print("Mean - StdDev: {0:.4f}").format(score.mean_validation_score - np.std(score.cv_validation_scores))
        print("Mean validation score: {0:.4f} (std: {1:.4f})".format(
              score.mean_validation_score, np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
        
        if params == None:
            params = score.parameters
    
    return params


def compare_scores(x, y):
    """
    Custom scoring function uses mean-stddev as the parameter set's score to give an extra bonus 
    to more consistent models
    """
    dx = x._asdict()
    dy = y._asdict()
    xstd = np.std(dx['cv_validation_scores'])
    ystd = np.std(dy['cv_validation_scores'])
    xscore = dx['mean_validation_score'] - xstd
    yscore = dy['mean_validation_score'] - ystd
    
    if abs(xscore - yscore) < 0.001:
        if abs(xstd-ystd) < 0.001:
            return 0
        elif xstd < ystd:
            return 1
        else:
            return -1
    elif xscore > yscore:
        return 1
    else:
        return -1