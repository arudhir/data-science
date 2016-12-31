import math
import numpy as np
import pandas as pd
import scipy as sp
import rpy2.robjects as robjects
import warnings
import sklearn
from copy import deepcopy
from rpy2.robjects.vectors import DataFrame
from rpy2.robjects.packages import importr, data
from rpy2.robjects import pandas2ri
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from scipy.interpolate import CubicSpline
from sklearn.metrics import confusion_matrix
from IPython.display import display, HTML
from collections import namedtuple
pandas2ri.activate()
biocinstaller = importr("BiocInstaller")
genefilter = importr("genefilter")
warnings.filterwarnings('ignore')

# %%
''' Returns how many elements in common a bicluster has with the list of progressed patients
does not return the bicluster with the most matches because there is no penalty for wrong answers
TODO: maybe combine with another algorithm '''
def biclusterCommon(clusters, progList):
    # max_val = -1
    # cluster_num = -1
    ret = []
    for i in range(clusters.n_clusters):
        n = np.sum((np.in1d(p, clusters.get_indices(i)[0])).astype(int))
        ret.append(n)
        #print(n)
        # if (max_val < n):
        #     max_val = n
        #     cluster_num = i
    return ret

''' Returns a data series of patients that have progressed'''
def progressedList(data):
    ret = []
    for i in data.index: # remember series are label indexed; our labels are numbers, so this can be confusing
        if (data[i]):
            ret.append(i+1) # Fix indicing at 0
    return pd.Series(ret)

'''Makes n confusion matrices and generates statistics corresponding to the true/false pos/negs
param: n_est (number of estimators for the randomforest), n_mat (number of matrices)
return: a tuple of statistics for each element of the confusion matrix'''
def confusionMatrixStatistics(train_data, truth_data, num_est, xy_cv, n_mat):

    X= xy_cv[0]
    y = xy_cv[1]["PROGRESSED"]

    confusion_matrix_arr = [confusion_matrix(y, classificationForest(train_data, data_norm.truth, num_est).predict(X)) for x in range(n_mat)]

    true_neg = [confusion_matrix_arr[i][0][0] for i in range(n_mat)]
    false_neg = [confusion_matrix_arr[i][1][0] for i in range(n_mat)]
    true_pos = [confusion_matrix_arr[i][1][1] for i in range(n_mat)]
    false_pos = [confusion_matrix_arr[i][0][1] for i in range(n_mat)]

    ret_tneg = sp.stats.describe(true_neg)
    ret_fneg = sp.stats.describe(false_neg)
    ret_tpos = sp.stats.describe(true_pos)
    ret_fpos = sp.stats.describe(false_pos)

    return (confusion_matrix_arr, (true_neg, false_neg, true_pos, false_pos), (ret_tneg, ret_fneg, ret_tpos, ret_fpos))

'''Prints and returns ranked features'''
def rankFeatures(forest, features):
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest], axis=0)
    indices = np.argsort(importances)[::-1]
    for f in range(features.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    return indices

'''Returns a forest of trees (classification). d = data tuple, n = number of estimators'''
def classificationForest(features, labels, n):
    training_val = alignData(features, labels)
    forest = RandomForestClassifier(n_estimators=n)
    X = training_val[0]
    y = training_val[1]["PROGRESSED"]
    ret = forest.fit(X, y)
    return ret

'''Returns a forest of trees (regression -- explicitly coded for TO values)'''
def regressionForest(features, labels, n):
    training_val = alignData(features, labels)
    forest = RandomForestRegressor(n_estimators=n)
    X = training_val[0]
    y = training_val[1]["TO"]
    ret = forest.fit(X, y)
    return ret


'''Takes two DataFrames and returns two versions of those DataFrames (tuple) but with the same rows in each'''
def alignData(df1, df2):
        index = (df1.index & df2.index)
        ret1 = df1.loc[index, :]
        ret2 = df2.loc[index, :]
        return (ret1, ret2)
# %%
def normalizeData(d):


    temp_exp = pd.DataFrame(sklearn.preprocessing.normalize(d.exp))
    temp_copy = pd.DataFrame(sklearn.preprocessing.normalize(d.copy))

    norm_TP = normWithNan(d.truth["TP"])
    norm_TO = normWithNan(d.truth["TO"])

    temp_truth = d.truth
    temp_truth["TP"] = d.truth["TP"].apply(lambda x: x / norm_TP)
    temp_truth["TO"] = d.truth["TO"].apply(lambda x: x / norm_TO)

    ret_data = Data(temp_exp, temp_copy, temp_truth)
    return ret_data
# %%
'''Finds the norm with data that has nan's'''
def normWithNan(v):

    sum = 0
    for elem in v:
        if (np.isnan(elem)):
            continue
        sum += (elem ** 2)
    ret = math.sqrt(sum)
    return ret

# %%
'''Cleans up the data by removing unusable na spots'''
def cleanData(d):
    temp_exp = d.exp.dropna(axis=0, how='all')
    temp_copy = d.copy.dropna(axis=0, how='all')

    temp_truth = d.truth.dropna(axis=0, how='all')

    temp_truth.insert(len(temp_truth.columns), "PROGRESSED", 0)
    temp_truth["PROGRESSED"] = ~(temp_truth["TP"].isnull())

    ret_data = Data(temp_exp, temp_copy, temp_truth)

    return ret_data

# %%
'''Implements Bioconductor's genefilter() on our gene data'''
def geneDataFilter(d):
    ffun = robjects.r("filterfun(cv(a = 0.7, b = 10))")

    # Transpose because I think genefilter wants genes in rows


    exp_temp = pd.DataFrame.transpose(d.exp)
    copy_temp = pd.DataFrame.transpose(d.copy)

    exp_r = pandas2ri.py2ri(exp_temp)
    copy_r = pandas2ri.py2ri(copy_temp)

    exp_filt = list(robjects.r.genefilter(exp_r, ffun))
    copy_filt = list(robjects.r.genefilter(copy_r, ffun))

    temp_exp = pd.DataFrame.transpose(pd.DataFrame.transpose(d.exp)[exp_filt])[1:]
    temp_copy = pd.DataFrame.transpose(pd.DataFrame.transpose(d.copy)[copy_filt])[1:]

    ret_data = Data(temp_exp, temp_copy, d.truth)

    return ret_data

# %%
'''Reads the csv files and returns it in a namedtuple called Data for readability
TODO: ADD MUTATION CSV'''
def readFiles(exp, copy, truth):
    Data = namedtuple('Data', 'exp copy truth', verbose=False)

    exp_csv = pd.read_csv(exp)
    copy_csv = pd.read_csv(copy)
    truth_csv = pd.read_csv(truth)


    ret_data = Data(exp_csv, copy_csv, truth_csv)

    return ret_data

if __name__ == '__main__':

    Data = namedtuple('Data', 'exp copy truth')
