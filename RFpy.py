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
from RFpyhelper import *
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
''' First, get the data. Second, split it into training and validation sets'''

data_init = readFiles("expressions_example.csv", "copynumber_example.csv", "mutations_example.csv", "groundtruth_example.csv")

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




1/7/16: Atom crashed, lost work on _10chik's solution. What was lost: data split, forest model creation, then making a mini test suite that returns results from roc_auc_score, f1_test, and confusion matrix — 10chik's solution does not work

also lost work on the 2 percent variance in mutation deletion thing, but that also did not work

todo: start cleaning up this motherfucker and find another way to do your work bc Atom keeps dicking you; maybe migrate to Jupyter proper, and combine code from RFpyhelper here perhaps? we'll do that if non-atom things also shit the bed
 '''

# Create the "is_unlabeled" label
data_init.truth.insert(len(data_init.truth.columns), "is_unlabeled", 0)
data_init.truth["is_unlabeled"] = np.isnan(data_init.truth["TO"])

temp1 = data_init.exp.add_suffix("_exp")
temp2 = data_init.copy.add_suffix("_copy")
temp3 = data_init.mut.add_suffix("_mut")

# We need to impute the NaN values before concatenating
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

imp.fit(temp1)
col_val = pd.Series(temp1.columns.values)
temp1 = pd.DataFrame(imp.transform(temp1))
temp1 = temp1.rename(columns=col_val)

imp.fit(temp2)
col_val = pd.Series(temp2.columns.values)
temp2 = pd.DataFrame(imp.transform(temp2))
temp2 = temp2.rename(columns=col_val)

imp.fit(temp3)
col_val = pd.Series(temp3.columns.values)
temp3 = pd.DataFrame(imp.transform(temp3))
temp3 = temp3.rename(columns=col_val)

temp4 = pd.concat([temp1, temp2], axis=1)
temp5 = pd.concat([temp4, temp3], axis=1)


forest = RandomForestClassifier(n_estimators=1000)

forfit = forest.fit(temp5, data_init.truth["is_unlabeled"])
importances = forfit.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(temp5.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



np.savetxt('feature importances', forfit.feature_importances_)

data_filt = geneDataFilter(data_init)
data_clean = cleanData(data_filt)
data_norm = normalizeData(data_clean) # whenever data_norm is changed, data_clean is also, idk whats going on
data_norm.exp.index = data_clean.exp.index # Somewhere the indices got messed up
data_norm.copy.index = data_clean.copy.index # But data_norm.truth is fine it seems
data_norm.truth.index = data_norm.truth.index + 1 # Since this started indexing at 0


''' Machine Learning Framework: http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/ '''

''' 1: Split data '''
# %%

mut_data = pd.read_csv("mutations_example.csv")
#a = a.add_suffix('_mut')
#b = b.add_suffix('_exp')


x = pd.concat([a, b], axis=1)
x


#d = data_norm.truth.iloc[0:4, 0:3]
#d.index = d.index + 1
#d = pd.concat([d, d], axis=0, join='inner')
d
forest = classificationForest(x, d, 10)
regforest = regressionForest(x, d, 10)
#pd.DataFrame(forest.feature_importances_, x.columns.values)

forest.feature_importances_
regforest.feature_importances_

# %%
# training a random forest classifier
# http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-7-3
train_data = data_norm.exp.sample(frac=0.5)
forest_reg_exp = regressionForest(train_data, data_norm.truth, 10)
forest_clss_exp = classificationForest(train_data, data_norm.truth, 10)

sklearn.ensemble.RandomForestClassifier.decision_path(data_norm.exp)
forest_clss_exp.estimators_

ranked_features = rankFeatures(forest_reg_exp, train_data)

imp_features = ranked_features[0:113]
norm_exp_imp_features = data_norm.exp.iloc[:, imp_features]
linear_model_data = alignData(norm_exp_imp_features, data_norm.truth)


linear_model_data[1]

linear_model = linear_model.Lasso(alpha=0.1)
linear_model.fit(linear_model_data[0], linear_model_data[1]["PROGRESSED"])




# %%
# cross validation, i.e., using the model we just trained
data_cv = data_norm.exp.sample(frac=0.25)
xy_cv = alignData(data_cv, data_norm.truth)

X_cv = xy_cv[0]
y_cv = xy_cv[1]["PROGRESSED"]

# %% Confusion Matrix
classificationForest(train_data, data_norm.truth, num_est).predict(X)
cv_confusion_matrix = confusion_matrix(y_cv, classificationForest(train_data, data_norm.truth, 10).predict(X_cv))



confusion_matrices = confusionMatrixStatistics(train_data, data_norm.truth, 10, xy_cv, 10)[0]
confusion_results = confusionMatrixStatistics(train_data, data_norm.truth, 10, xy_cv, 10)[1]
confusion_statistics = confusionMatrixStatistics(train_data, data_norm.truth, 10, xy_cv, 10)[2]


# %% ROC AUC score
# TODO: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
metrics.roc_auc_score(y_cv, classificationForest(train_data, data_norm.truth, 10).predict(X_cv))
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_cv, classificationForest(train_data, data_norm.truth, 10).predict(X_cv))
fpr
tpr
thresholds
# %% Hazard and Survival Functions
# https://lifelines.readthedocs.io/en/latest/
orig_truth_clean = cleanData(data_filt).truth # Non-normalized data just bc idk what normalizing does to KM graphs
T = orig_truth_clean["TP"]
C = orig_truth_clean["PROGRESSED"]

# Survival Function
# I think survival functions make the most sense in the context of having two (maybe more) different groups of things to plot; right now we only have one.
# Example: Gene A is mutated in some, not mutated in others --> we can make a KM plot to see if the mutation (or lack thereof) leads to dec survival
# Otherwise, rn, all we get is "survival decreases over time". no fucking shit.



kmf = KaplanMeierFitter()
# Get the event times for each sample
#orig_truth_clean.insert(len(orig_truth_clean.columns), "EVENT_TIMES", 0)
#orig_truth_clean["EVENT_TIMES"] = orig_truth_clean["TO"].subtract(orig_truth_clean["TP"], fill_value=0)

kmf.fit(T, C) # Not sure if we should use the original/train/or CV data
kmf.survival_function_
kmf.median_

plt.title('Survival function of multiple myeloma patients', axes=kmf.plot()) # Plotted with confidence intervals


# Cumulative Hazard Function
naf = NelsonAalenFitter()
naf.fit(T, event_observed=C)
plt.title('Hazard function of multiple myeloma patients', axes=naf.plot())

# %% Survival regression

# In order to do Cox regression we need to first find a way to create a covariance matrix with our high-dimensional feature space
# Typical examples only have a reasonable (see: not 57000) features, and therefore it is trivial to write a linear regression model for those
# Our relationships are probably more complex and definitely impossible to write out by hand

# https://en.wikipedia.org/wiki/Proportional_hazards_model#Under_high-dimensional_setup


# Therefore, the way to go is probably LASSO first

# or do logistic regression and call it a fucking day

# Idea: Group LASSO on the gene signature EMC-92 associated with multiple myeloma

















# %% Biclustering to get features
# experimental
clusters = sklearn.cluster.bicluster.SpectralCoclustering(n_clusters = 50)
fitted_clusters = clusters.fit(data_norm.copy)
pList = pd.DataFrame.as_matrix(progressedList(data_norm.truth["PROGRESSED"]))
biclusterCommon(fitted_clusters, pList)[40]
fitted_clusters.get_indices(40)[1]
type(clusters)
pd.DataFrame.as_matri
# %%
forest = RandomForestClassifier()

'''
HELPER FUNCTIONS
'''

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
    '''
    exp_csv = pd.read_csv(exp)
    copy_csv = pd.read_csv(copy)
    truth_csv = pd.read_csv(truth)
    mut_csv = pd.read_csv(mut)
    ret_data = Data(exp_csv, copy_csv, mut_csv, truth_csv)

    return ret_data
