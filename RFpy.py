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
import seaborn as sns
from rpy2.robjects.vectors import DataFrame
from rpy2.robjects.packages import importr, data
from rpy2.robjects import pandas2ri
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, normalize, Imputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score
from sklearn.manifold import Isomap
from sklearn.covariance import graph_lasso, GraphLassoCV, ledoit_wolf, EmpiricalCovariance
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
from lifelines.utils import datetimes_to_durations, survival_table_from_events, k_fold_cross_validation
from lifelines import AalenAdditiveFitter, CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test
from lifelines.datasets import load_rossi
from scipy.interpolate import CubicSpline
from sklearn.externals import joblib
from IPython.display import display, HTML
from collections import namedtuple
from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass
%matplotlib inline
%pylab inline
warnings.filterwarnings('ignore')
pandas2ri.activate()
biocinstaller = importr("BiocInstaller")
genefilter = importr("genefilter")
#pcaMethods = importr("pcaMethods") # not used yet; $ conda install bioconductor-pcamethods

# %%
'''
A little bit of processing before the preprocessing.
'''
# Data = namedtuple('Data', 'exp cop mut labels')
exp = pd.read_csv("expressions_example.csv").add_suffix("_exp")
cop = pd.read_csv("copynumber_example.csv").add_suffix("_cop")
mut = pd.read_csv("mutations_example.csv").add_suffix("_mut")
labels = pd.read_csv("groundtruth_example.csv")

addLabel(labels, "PROGRESSED")
nan_indices = labels["TO"].index[labels["TO"].apply(np.isnan)]
labels["PROGRESSED"] = ~labels["TP"].isnull()
labels["PROGRESSED"].ix[nan_indices] = np.nan

# From here, modifications to data will be context-specific.
exp.to_csv("exp_suffix.csv")
cop.to_csv("cop_suffix.csv")
mut.to_csv("mut_suffix.csv")
labels.to_csv("labels_suffix.csv")

'''
Preprocessing/Data Splitting
============================
Handle NaN values for sklearn ML functions; handling of NaN values will depend on the situation. Currently, NaN values are being handled by imputing the NaN values with the mean of the column.

Then, combine features into variable X, and assign variable y to the labels.

Then, split the data into the training set and the validation set. Note: I don't think there is enough data to split the samples into a training, cross-validation (sometimes called validation), and testing set. k-fold cross-validation handles this situation. The basic approach involves initially splitting the data into two parts: training set and testing set. However, instead of using a validation set — split the training set into k smaller sets ("folds"), and train the model using (k-1) folds, and validate the model using the remaining fold, then evaluate the performance by some metric. Then run this in a loop and average all the metrics. Although computationally expensive — data will be saved.. Additional notes:
    - Use stratified k-fold cross-validation, because this handles situations where label distribution is heavily skewed, and ensures that relative class frequencies between folds is preserved.
'''
# When we restart the kernel just load the matrices we sent to csv for convenience/for the sake of memory.
exp = pd.read_csv("exp_suffix.csv")
cop = pd.read_csv("cop_suffix.csv")
mut = pd.read_csv("mut_suffix.csv")
labels = pd.read_csv("labels_suffix.csv")

# Impute the missing values - a copy will be made here
exp_nonan = featureImputer("NaN", "mean", 0, exp)
cop_nonan = featureImputer("NaN", "mean", 0, cop)
#mut_impute = featureImputer("NaN", "most_frequent", 0, Data_suffix.mut) # Perhaps using the mean makes sense too — assigning a probability that a gene is mutated given the rest
mut_nonan = featureImputer("NaN", "mean", 0, mut)

# Eliminate the NAN values from the labels for now — see if there is a better way of handling this; proportional hazards model might deal with this better
labels_nonan = labels.copy()
nan_indices = labels["PROGRESSED"].loc[np.isnan(labels["PROGRESSED"])].index
mask = labels_nonan["PROGRESSED"].index.isin(nan_indices)
labels_nonan = labels_nonan[~mask]

# Combine all the features; I think this is referred to as "feature stacking"
X = pd.concat([exp_nonan, pd.concat([cop_nonan, mut_nonan], axis=1)], axis=1)
y = labels_nonan["PROGRESSED"] # Classification labels

# Make sure there is a label for every sample in X
X, y = alignData(X, y)


'''
Get our two sets — stratify wrt y to make sure there is an equal proportion of progressed/not progressed. From here, we can start applying various algorithms and transformations.

Pipelines should simplify combining various kinds of transformations.
'''
# TODO: Make sure there is a different X_train, test, y, etc. for each model
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size = 0.25, stratify=y)
y_train = np.ravel(y_train) # Documentation for the fitting functions gives a warning when using a dataframe
y_test = np.ravel(y_test)

'''
Dimensionality Reduction
========================
Reduce the dimensionality of the data. There are primarily two approaches: eigenvalue decomposition methods (e.g., PCA) or discriminant-based methods (e.g., LDA)

1/9/17: Skipping this because it's not needed for an inital model.
'''
# TODO: In order to generate many PCA estimators for finding the optimal parameter, write a wrapper method to generate PCA's with varying n's
# Remember PCA is projecting the data on a lower-dimensional axis, mixing all features to find the orthogonal directions of maximum variance.
pca = PCA() # pca.components_ returns the same shape as X_train; defaults to projecting to min(n_samples, n_features)-dimension space.
X_pca = pca.fit_transform(X_train) # dimension is num_samp x num_samp; defaults as min(num_feat, num_samp)

# Pickle the transformer
joblib.dump(pca.transform(X_train), 'pcaTransform.pkl')

'''
Feature Selection
=================
Select the most predictive features from the data.

Note: joblib.dump(transformer, 'filename.pkl') to store transformers or a pipeline in order to apply them to the test set later.

NO NEED: Normalize the expression data.
    - Per forum post: Gene expression data: values are normalized by FPKM (fragment per kilobase per million).
                      Copy number data: values represent the log-2 scaled ratio between tumor and normal tissue. 0 means no copy change, negative values are deletions, and positive values are amplifications.
Q: Aren't we losing data we can't afford to lose when excluding ~80 people from being included in the feature selection process?
    TODO: See how different the result is on the whole dataset vs. just the training sample. Get the genes that are scored >0.0 in both, make them into Sets, and use logical&

Note: The copy number and/or the mutation data might just be added noise — most of the papers deal with just expression data. Find a more intelligent way to incorporate them, or eliminate them.

Iterative Pearson correlation coefficients: Generates Pearson correlation coefficients in the initial m samples x n features feature space to get an mxm matrix, which is then interpreted as m-samples m-features, finds the Pcc again, and iterates; reveals sample similarity hiding in higher-order correlation features, enlarge weak patterns underlying the raw dataset, and preserve the balanced independent relationship.
http://nar.oxfordjournals.org/content/early/2013/06/12/nar.gkt343.full
    • Doesn't seem hard to implement and since all of the patients have myeloma, this might be a better way to find finer differences between the patients — ones that would lead to progression time differences.
    • Maybe a NN can be made using this? dunno
    • The accuracy of both k-means clustering and naive Bayes jumped like 15% when just using first-order coefficients.
    • Sample workflow: Select top 1000 features using F-test, then iPcc
    • "The reduced dimensionality allows rapid and accurate computation of the global optimum of many clustering and classification algorithms and thus improves accuracy." -- maybe this can be combined with
    • "However, iPcc is not an independent algorithm for disease class discovery or prediction. It provides an effective means to underpin the underlying patterns embedded within the gene expression data sets from the feature extraction perspective. Therefore, it can be used in combination with other clustering, classification, feature selection and feature extraction algorithms, as demonstrated in the results section."
    • Can't find a wrapper function for the iPcc, maybe I can make one and give it to scikit-learn
        • I think this might be called something else — "hierarchal clustering", or maybe we can use existing clustering algorithms iteratively with the Pearson correlation coefficient as the metric.
Instead of approaching this problem as finding features to predict cancer (because they already all have cancer), perhaps approaching it from a cancer subtyping problem (Ren X, Wang Y, Wang J, Zhang XS. A unified computational model for revealing and predicting subtle subtypes of cancers. BMC Bioinformatics 2012;13:70.) might be fruitful, where the subtypes are how quickly they progress.
'''

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
feat_imp = rankFeatures(clf, X_train)
# selected_feature_indices = pd.Series(feat_imp[:X_new.shape[1]])

rf_model = SelectFromModel(clf) # Not preserving column names or index names
X_rf_trans = rf_model.fit_transform(X_train, y_train)


# TODO: Make the below into a helper function because this obnoxious
# Transform, cast to DF, find the top features. cast that to a series, find its columns, and rename the DF's columns accordingly.
# https://github.com/scikit-learn/scikit-learn/issues/6425
# The idea of preserving feature names across transformers (a get_feature_names) is an ongoing issue that is still unresolved
# Thinking about it again, there doesn't seem to be a need to save feature names (unless I am personally curious)
# X_new = pd.DataFrame(model.transform(X_train)).rename(columns=pd.Series(X_train[feat_imp[:X_new.shape[1]]].columns)) # columns
# X_new.index = y_train.index # indices

# Pickle the transformer
joblib.dump(rf_model.transform(X_train), 'clf_rf.pkl')




'''
Hyper-parameter Selector Methods
================================
Optimize the hyper-parameters in the various models/methods called in Dimensionality Reduction, Feature Selection, and Model Selection

Hyper-parameters are parameters that are not directly learnt within estimators, but are passed as arguments to the constructors of the estimator class. Think of the alpha parameter in LASSO — it's the stuff we pick.

This may have to be organized as a specific section for each context.

Again, use k-fold cross-validation in combination with sklearn's grid search capabilities to tune hyper-parameters.
http://scikit-learn.org/stable/modules/grid_search.html

Look into learning curves, bias and variance curves, and the other stuff taught in Coursera.
'''
# Pipeline all the steps so far
# Note: calling fit() on the pipeline does the same thing as fitting, then transforming for the next thing to fit it.
estimators = [('pca', PCA()), ('clf', RandomForestClassifier())]
pipe = Pipeline(estimators)

# pipe.fit(X_train, y_train)
# Pick the parameters you want to vary and their values
params = dict(pca__n_components=[10, 20, 40, 60, 80, 100, X_train.shape[1]],
         clf__n_estimators=[10, 25, 50, 100, 500],
         clf__max_features=['sqrt', 'log2', None],
         clf__criterion=['entropy'])

grid_search = GridSearchCV(pipe, param_grid=params, cv=10, scoring='roc_auc')

# Fit the training data to get the best parameters for it — I don't think we would need to fit the PCA() and RandomForestClassifier() separately then.
grid_search.fit(X_train, y_train)


'''
Model Selection
===============
Make a model.

Look into the idea of ensembling a bunch of models (this might be called boosting)

Look into a Bayesian network to get probabilities as well as k-means

'''


# Modify the X_test data appropriately
X_test_trans = pipe.fit_predict(X_test)


# Logistic Regression
lr = LogisticRegression()
lr.fit(X_rf_trans, np.ravel(y_train)) # does not include PCA; need to pipeline this beter later

# Don't forget to transform the data
X_lr_trans = model.fit_transform(X_train, y_train)


# X_test_trans = X_test[selected_feature_indices]
# y_pred = lr.predict(X_test_trans)

X_lr_trans.predict(y










SelectFromModel(lr

estimators = [('reduce_dim', PCA(),

estimators = [('reduce_dim', PCA()), ('lr', LogisticRegression()), ('clf', RandomForestClassifier())]
pipe = Pipeline(estimators)
scores = cross_val_score(pipe, X_train, np.ravel(y_train), scoring="roc_auc", cv=10) # Internally creates a cross-validation generator using the integer passed as the number of (stratified) K folds. This really makes it a lot easier, lol.

scores






'''
Model Evaluation
================
Gives metrics about the predictive accuracy of the model.

TODO: Bias/variance decomposition to see where the inaccuracy lies
    There may not be a need though because its not like you have an option to get more samples.
'''

# The logistic regression model without tuned hyperparameters, or optimized anything (but with feature selection) is 5% better than flipping a coin.
confusion_matrix(y_test, y_pred)
roc_auc_score(y_test, y_pred) # 0.55147058823529405
f1_score(y_test, y_pred) # 0.31578947368421056



'''
Survival Analysis
=================
Look into Gaussian Processes as well

3 approaches to fitting survival models (http://data.princeton.edu/wws509/notes/c7s3.html):
    1) Parametric
    2) Semi-parametric
    3) Non-parametric
        - Cox. Functions by leaving the baseline hazard unspecified and relies on a partial likelihood function.

Although I think Cox is the approach that should be taken, it doesn't seem to be working the way I am using it. Read up on how Cox is used in the context of disease progression and genomics.

"SURVIV for survival analysis of mRNA isoform variation" (2016) - http://www.nature.com/articles/ncomms11548
    • "Simulation studies suggest that SURVIV outperforms the conventional Cox regression survival analysis, especially for data sets with modest sequencing depth."
    ° just kidding, this is for RNAseq data. probably has links to example survival analyses though

'''

# TODO: Figure out the mathematics behind Cox Regression and how to implement it properly.
cf = CoxPHFitter()
labels_train = labels.iloc[y_train.index]
surviv_mat = pd.concat([labels_train.iloc[:, 2:4], X_new], axis=1)
cf.fit(surviv_mat, "TO", "PROGRESSED", include_likelihood=True) # adding either include_likelihood or strata made predict_cumulative_hazard work -- find out why

cox_surv_func = cf.predict_survival_function(X_test)
cf.predict_cumulative_hazard(X_new) # also gives everone the same number
cf.predict_expectation(X_new) # gives everyone the same number
cf.print_summary()
cf.baseline_hazard_
cf.baseline_survival_

lifelines.utils.concordance_index(

cox.baseline_survival_.plot()
cox.baseline_hazard_.plot()
cox.baseline_cumulative_hazard_.plot()


# rossi_dataset = load_rossi()
# cf = CoxPHFitter()
# cf.fit(rossi_dataset, 'week', event_col='arrest')
# cf.print_summary()
'''
TODO: 1/8/17
- Figure out how to impute a boolean matrix — DONE
- Take some time to see if xgboost can deal with sparse matrices so you don't have to make questionable decisions with the data — DONE
    - ANS: Maybe not, because NaN elements are not zero and cannot be considered as such! We have a dense matrix with missing/censored values.
- Split the data - DONE
- Stack features — DONE


TODO: 1/9/17
- Pipeline all of the things you can using sklearn.pipeline.Pipeline
- Throw it all into a tree-based model to feature select - DONE
- Make a ML model - DONE
- Optimize hyperparameters with the CV set
- Make some functions to visualize how good the ML model is — maybe learn seaborn or something.
- Fix the memory issue that's happening - MAYBE; I think it's Jupyter/Atom shitting it up not the code
- Figure out how to use sklearn.pipeline.Pipeline to pipeline the transformation steps for ease
    - Think about whether or not this is possible with the fact that feature names don't seem to get preserved across transformations
    - Make a wrapper function to address the feature name issue with transformations.

TODO: 1/10/17
    • Start up various pipelines to organize transformations
        • Try and see if we need to keep feature names at all, or a way to relate y_values and the X changes
        •

Note on sklearn.pipeline.Pipeline - Allows for a convenient way to apply a fixed sequence of steps on our data, e.g., say we have to feature select, normalize, and classify. Pipeline would allow us to only have to call fit() and transform() once on our data, and allow us to use grid search (check sklearn docs) over all all estimators in the pipeline at once. A pipeline consists of estimators, all of which (except the last one) have to be transformers (i.e., have a transform method). The last one can be whatever.
'''

# There is a memory problem here - I think over 600mb is being allocated... Unsure if this is because of Jupyter or because of me.
# TODO: Allocate memory better - do more inplace dataframe making, no need to save onto the old things if we write to a csv file anyways. -- DONE, I think
# total_size(Data_nonan, verbose=True) + total_size(X) + total_size(y) + total_size(Data_suffix) + total_size(Data_init) = 602346560 bytes
# Solution: Dump to file so that we can access later

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
        sklearn.ensemble.forest.RandomForestClassifier
            forest: an ensemble of decision trees for classification
        list
            features: The attribute feature_importances_ associated with a fitted forest.
    Returns: An np.array sorted by feature importance.
    '''
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest], axis=0)
    indices = np.argsort(importances)[::-1]
    # for f in range(features.shape[1]):
    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
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
        df1 numpy ndarray; a dict with pandas.Series, arrays, constants, or list-like values; pandas.DataFrame
        df2 numpy ndarray; a dict with pandas.Series, arrays, constants, or list-like values; pandas.DataFrame
    Return: A tuple of DataFrames with the same row indices.

    Also tries to be fancy with exception handling.
    Q: Are assignments in try blocks frowned upon? What if you can't assign? Do I add another try/except?
    '''
    if not isinstance(df1, pd.DataFrame):
        df1 = pd.DataFrame(df1) # There is not a scope issue because df1 and df2 are passed parameters.
    if not isinstance(df2, pd.DataFrame):
        df2 = pd.DataFrame(df2)

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
