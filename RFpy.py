# %%
import math
import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk
import rpy2.robjects as robjects
import warnings
import matplotlib.pyplot as plt
import lifelines
from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass
from rpy2.robjects.vectors import DataFrame
from rpy2.robjects.packages import importr, data
from rpy2.robjects import pandas2ri
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler, normalize, Imputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.manifold import Isomap
from sklearn.covariance import graph_lasso
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedKFold
from lifelines.utils import datetimes_to_durations, survival_table_from_events
from lifelines import AalenAdditiveFitter, CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test
from scipy.interpolate import CubicSpline
from IPython.display import display, HTML
from collections import namedtuple
%matplotlib inline
warnings.filterwarnings('ignore')
pandas2ri.activate()
biocinstaller = importr("BiocInstaller")
genefilter = importr("genefilter")
#pcaMethods = importr("pcaMethods") # not used yet; $ conda install bioconductor-pcamethods

# %%
'''
A little bit of processing before the preprocessing. Create a namedtuple Data to organize csv data, add suffixes to distinguish features, and add the PROGRESSED label.
'''
Data = namedtuple('Data', 'exp cop mut labels')
file1 = pd.read_csv("expressions_example.csv")
file2 = pd.read_csv("copynumber_example.csv")
file3 = pd.read_csv("mutations_example.csv")
file4 = pd.read_csv("groundtruth_example.csv")
Data_init = Data(file1, file2, file3, file4)

Data_suffix = addSuffixes(Data_init)
addLabel(Data_suffix.labels, "PROGRESSED")
nan_indices = Data_suffix.labels["TO"].index[Data_suffix.labels["TO"].apply(np.isnan)]
Data_suffix.labels["PROGRESSED"] = ~Data_suffix.labels["TP"].isnull()
Data_suffix.labels["PROGRESSED"].ix[nan_indices] = np.nan # Otherwise False shows up instead of NAN

# del file1, file2, file3, file4, Data_init
Data_suffix.exp.to_csv("exp_suffix.csv")
Data_suffix.cop.to_csv("cop_suffix.csv")
Data_suffix.mut.to_csv("mut_suffix.csv")
Data_suffix.labels.to_csv("lab_suffix.csv")
'''
Since sklearn cannot handle NaN values, we will either 1) Impute NaN values or 2) Eliminate bad rows, depending on the situation. Afterwards, split the data into the training set, the cross validation set, and the test set in a 50/25/25 split.
'''

# Impute the missing values
exp_impute = featureImputer("NaN", "mean", 0, Data_suffix.exp)
cop_impute = featureImputer("NaN", "mean", 0, Data_suffix.cop)
#mut_impute = featureImputer("NaN", "most_frequent", 0, Data_suffix.mut) # Perhaps using the mean makes sense too — assigning a probability that a gene is mutated given the rest
mut_impute = featureImputer("NaN", "mean", 0, Data_suffix.mut)

# Eliminate the NAN values from the labels for now — see if there is a better way of handling this; proportional hazards model might deal with this better
label_nonan = Data_suffix.labels.copy()
mask = label_nonan["PROGRESSED"].index.isin(nan_indices)
label_nonan = label_nonan[~mask]

# Make a Data tuple with non-NAN data
Data_nonan = Data(exp_impute, cop_impute, mut_impute, label_nonan)



# Split the data
# Concatenate all the data dataframes to make one big X
X = pd.concat([Data_nonan.exp, pd.concat([Data_nonan.cop, Data_nonan.mut], axis=1)], axis=1)
y = Data_nonan.labels["PROGRESSED"] # Classification labels

eval_size = 0.10
kf = StratifiedKFold(y, round(1. / eval_size))
train_indices, valid_indices = next(iter(kf))
X_train, y_train = X[X.index[train_indices]], y[y.index[train_indices]] # kf returns the indices of df.index, not df - therefore we need to change back to our dataframe's index
X_valid, y_valid = X[X.index[valid_indices]], y[y.index[valid_indices]]

# There is a memory problem here - I think over 600mb is being allocated... Unsure if this is because of Jupyter or because of me.
# TODO: Allocate memory better - do more inplace dataframe making, no need to save onto the old things if we write to a csv file anyways.
# total_size(Data_nonan, verbose=True) + total_size(X) + total_size(y) + total_size(Data_suffix) + total_size(Data_init) = 602346560 bytes
# Solution: Dump to file so that we can access later


'''
TODO: 1/8/17
- Figure out how to impute a boolean matrix — DONE
- Take some time to see if xgboost can deal with sparse matrices so you don't have to make questionable decisions with the data — DONE
    - ANS: Maybe not, because NaN elements are not zero and cannot be considered as such! We have a dense matrix with missing/censored values.
- Split the data - DONE
- Stack features — DONE
- Throw it all into a tree-based model to feature select
- Make a ML model
- Optimize hyperparameters with the CV set
- Make some functions to visualize how good the ML model is — maybe learn seaborn or something.
- Fix the memory issue that's happening.

'''
# %%
'''
HELPER FUNCTIONS
'''
def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    Taken from https://code.activestate.com/recipes/577504/

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

def featureImputer(mv, strat, ax, f):
    '''
    Imputes the NaN values for the feature set passed, transforms the dataframe, and makes sure the columns are what they ought to be.

    Note: Imputer creates a copy internally.

    Args:
        mv (str): The type of feature Imputer finds.
        strat (str): Strategy for Imputer.
        ax: Axis for Imputer.
        f (pd.DataFrame): A feature dataframe
        *f_add: Any additional features.
    Return:

    '''
    if f is None:
        raise TypeError

    imp = Imputer(missing_values=mv, strategy=strat, axis=ax)

    col_val = pd.Series(f.columns.values)
    f = pd.DataFrame(imp.fit_transform(f))
    f = f.rename(columns=col_val)
    return f

def addSuffixes(d):
    '''
    Adds suffixes to our Data. Returns a completely new Data tuple.
    '''
    te = d.exp.add_suffix("_exp")
    tc = d.cop.add_suffix("_copy")
    tm = d.mut.add_suffix("_mut")
    tl = d.labels.copy()
    ret = Data(te, tc, tm, tl)
    return ret

def addLabel(self, new_label):
    '''
    Inserts a new label — modifies labels, does not create copy. Initializes to 0
    Args:
        pd.DataFrame of labels
        str of the new label
    Return:
        None
    '''
    self.insert(len(self.columns), new_label, 0)

def biclusterCommon(clusters, progList):
    '''
    Finds the elements that are the same between a bicluster and a list of progressed patients.

    Args:
        clusters (sk.cluster.bicluster.SpecteralCoclustering): A group of biclusters
        progList (numpy array): array of progression status of patients

    Returns:
        List: The shared elements
    '''
    ret = []
    for i in range(clusters.n_clusters):
        n = np.sum((np.in1d(p, clusters.get_indices(i)[0])).astype(int))
        ret.append(n)
    return ret

def progressedList(data):
    '''
    Gives the patients who have progressed.

    Args:
        data (pandas.DataFrame): Ground truths

    Returns:
        pandas.Series of patients who have progressed only
    '''
    ret = []
    for i in data.index: # remember series are label indexed; our labels are numbers, so this can be confusing
        if (data[i]):
            ret.append(i+1) # Fix indicing at 0
    return pd.Series(ret)

def confusionMatrixStatistics(train_data, truth_data, num_est, xy_cv, n_mat):
    '''
    Summarizes confusion matrix statistics for a specified number of classification forests.

    Args:
        train_data (pd.DataFrame): The data that will be used to train the classification forest.
        truth_data (pd.DataFrame): The associated labels for the training data.
        num_est (int): The number of estimators for our classification forest.
        xy_cv: A tuple of the cross validation data and its labels.
        n_mat: The number of confusion matrices we want to create.

    Returns:
            tuple: A tuple containing the list of confusion matrices, a list of the elements in each confusion matrix, and a list of the associated statistics.
    '''
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

def rankFeatures(forest, features):
    '''
    Prints forest features ranked by their importance.

    Args:
        forest (sklearn.ensemble.forest.RandomForestClassifier or RandomForestRegressor): An ensemble of decision trees.
        features (array): The attribute feature_importances_ associated with a fitted forest.

    Returns: An np.array sorted by feature importance.
    '''
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest], axis=0)
    indices = np.argsort(importances)[::-1]
    for f in range(features.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    return indices

def classificationForest(features, labels, n):
    '''
    Creates and fits a random forest classifier on given features and labels.

    Note: This specifically looks at the PROGRESSED column of the ground truth dataframe.

    Args:
        features: array-like or sparse matrix of shape = [n_samples, n_features]
        labels: pandas DataFrame of ground truths.
        n = number of estimators (trees in the forest)
    Returns:
        A classification forest of trees from the training set (features, labels)
    '''
    training_val = alignData(features, labels)
    forest = RandomForestClassifier(n_estimators=n)
    X = training_val[0]
    y = training_val[1]["PROGRESSED"]
    ret = forest.fit(X, y)
    return ret

def regressionForest(features, labels, n):
    '''
    Creates and fits a random forest regressor on given features and labels.

    Note: This specifically looks at the TO column of the ground truth dataframe.

    Args:
        features: array-like or sparse matrix of shape = [n_samples, n_features].
        labels: pandas DataFrame of ground truths.
        n = number of estimators (trees in the forest).
    Returns:
        A regression forest of trees from the training set (features, labels)
    '''
    training_val = alignData(features, labels)
    forest = RandomForestRegressor(n_estimators=n)
    X = training_val[0]
    y = training_val[1]["TO"]
    ret = forest.fit(X, y)
    return ret

def alignData(df1, df2):
    '''
    Create dataframes that have the same row indices (same samples)

    Args:
        df1 (pandas.DataFrame)
        df2 (pandas.DataFrame)
    Return: A tuple of DataFrames with the same row indices.
    '''
    index = (df1.index & df2.index)
    ret1 = df1.loc[index, :]
    ret2 = df2.loc[index, :]
    return (ret1, ret2)

def normalizeData(d):
    '''
    Normalizes expression, copy, and ground truth data.

    DEPRECIATED: does not handle mutation data. Do not use.
    '''

    temp_exp = pd.DataFrame(sklearn.preprocessing.normalize(d.exp))
    temp_copy = pd.DataFrame(sklearn.preprocessing.normalize(d.copy))

    norm_TP = normWithNan(d.truth["TP"])
    norm_TO = normWithNan(d.truth["TO"])

    temp_truth = d.truth
    temp_truth["TP"] = d.truth["TP"].apply(lambda x: x / norm_TP)
    temp_truth["TO"] = d.truth["TO"].apply(lambda x: x / norm_TO)

    ret_data = Data(temp_exp, temp_copy, temp_truth)
    return ret_data

def normWithNan(v):
    '''
    Finds the norm of data that has nan elements.

    Args:
        pandas Series
    Return:
        int: The l2 norm
    '''
    sum = 0
    for elem in v:
        if (np.isnan(elem)):
            continue
        sum += (elem ** 2)
    ret = math.sqrt(sum)
    return ret

def cleanData(d):
    '''
    Drops rows that are entirely na's while adding the PROGRESSED column to the ground truth dataframe.

    TODO: Figure out how to create a deep copy Data to preserve the original.
    DEPRECIATED: No functionality for mutations and is generally a poor function. Do not use.
    Args:
        namedtuple Data.
    Return:
        namedtuple Data that has been appropriately modified.
    '''
    temp_exp = d.exp.dropna(axis=0, how='all')
    temp_copy = d.copy.dropna(axis=0, how='all')

    temp_truth = d.truth.dropna(axis=0, how='all')

    temp_truth.insert(len(temp_truth.columns), "PROGRESSED", 0)
    temp_truth["PROGRESSED"] = ~(temp_truth["TP"].isnull())

    ret_data = Data(temp_exp, temp_copy, temp_truth)

    return ret_data

def geneDataFilter(d):
    '''
    Implements the Bioconductor function genefilter() to remove genes based on coefficient of variation.

    Args:
        namedtuple Data
    Return:
        modified namedtuple Data
    '''
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

def readFiles(exp, copy, mut, truth):
    '''
    Read in the initial csv files and return a namedtuple Data of the csv files.

    TODO: Figure out why there is a positional argument issue with this; ret_data line sees 5 arguments. Suspecting "self" is the cause — but this worked fine in the past.
    '''
    exp_csv = pd.read_csv(exp)
    copy_csv = pd.read_csv(copy)
    truth_csv = pd.read_csv(truth)
    mut_csv = pd.read_csv(mut)
    ret_data = Data(exp_csv, copy_csv, mut_csv, truth_csv)

    return ret_data

'''
NOTES AND RAMBLINGS
'''

'''
Current issue: Our groundtruth data is sparse — the most ideal way to proceed is by using semi-supervised learning. Liang et. al
(2016) describe a method for cancer survival analysis using semi-supervised learning with quasinorm regularization.

Q: Can we proceed using L_1 regularization? Or do we need to implement L_(1/2) regularization to follow the paper?
Q: Given that there are no libraries for quasinorm regularization (presumably the coordinate descent algorithm needs to be
modified as well) — is it feasible to write one? How would we test the package?
Q: If we decide to go with L_1 for now, is the appropriate action to get rid of all absent patient data? Is this even worth doing
given that we would drop 26 out of 332 entries (~7%)?
    Suggested from #bioinformatics: Introduce a new label (is_unlabeled) and fit a model to that. If a predictive model is found (i.e., AUC > 0.8), then drop the features used to make that model (i.e., the features that are (loosely) 'unique' to the unlabeled, and then drop the unlabeled samples and go from there.

Q: Where do control samples factor into this? Do I need to go out and find control examples? Should I?

Model Selection: Selecting a statistical model from a set of candidate models, given data. In the context of machine learning, this is referred to as 'hyperparameter optimization', where the usual goal is to optimize a measure of the algorithm's performance on an independent dataset. We use _cross validation_ to estimate the generalization performance of this. This is different than actual learning problems, which optimize based on a loss function in order to learn parameters that model the input well, whereas hyperparameter optimization is to make sure the model doesn't overfit.

Suggested from _10chik on #bioinformatics (freenode): Introduce a new label (is_unlabeled) and fit a model to that. If a predictive model is found (i.e., AUC > 0.8), then drop the features used to make that model (i.e., the features that are (loosely) 'unique' to the unlabeled, and then drop the unlabeled samples and go from there.
    From here, would it then make sense to introduce the dropped features one by one to see how they impact the overall score?
    What if the genes I drop are really important? I guess they'd be in the other patients too if they were THAT important
    ********** update 1/7/17: this doesn't work. neither does the stats.stackexchange proposal. **********

1/7/16: Atom crashed, lost work on _10chik's solution. What was lost: data split, forest model creation, then making a mini test suite that returns results from roc_auc_score, f1_test, and confusion matrix — 10chik's solution does not work, neither does stats.stackexchange solution. recode these for completeness.

'''
