import numpy as np
from operator import itemgetter

# Utility function to report optimal parameters
def report(grid_scores, n_top=20):
    print grid_scores
    
    params = None
    #top_scores = sorted(grid_scores, cmp=compare_scores, reverse=True)[:n_top]
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

# Use mean-stddev as the parameter set's score to give extra bonus to less variable models
def compare_scores(x, y):
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