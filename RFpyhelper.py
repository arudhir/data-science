import math
import numpy as np
import pandas as pd
import scipy as sp
import rpy2.robjects as robjects
import warnings
import sklearn
from RFpyhelper import *
from rpy2.robjects.vectors import DataFrame
from rpy2.robjects.packages import importr, data
from rpy2.robjects import pandas2ri
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn import svm
from scipy.interpolate import CubicSpline
from IPython.display import display, HTML
from collections import namedtuple
pandas2ri.activate()
biocinstaller = importr("BiocInstaller")
genefilter = importr("genefilter")
warnings.filterwarnings('ignore')

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
'''Reads the csv files and returns it in a namedtuple called Data for readability'''
def readFiles(exp, copy, truth):
    Data = namedtuple('Data', 'exp copy truth', verbose=False)

    exp_csv = pd.read_csv(exp)
    copy_csv = pd.read_csv(copy)
    truth_csv = pd.read_csv(truth)

    ret_data = Data(exp_csv, copy_csv, truth_csv)

    return ret_data

if __name__ == '__main__':

    Data = namedtuple('Data', 'exp copy truth')
