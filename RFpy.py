# %%
import math
import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk
import rpy2.robjects as robjects
import warnings
import sklearn
import matplotlib.pyplot as plt
import lifelines
from RFpyhelper import *
from rpy2.robjects.vectors import DataFrame
from rpy2.robjects.packages import importr, data
from rpy2.robjects import pandas2ri
from sklearn.preprocessing import MinMaxScaler, normalize
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
pcaMethods = importr("pcaMethods") # not used yet; $ conda install bioconductor-pcamethods

# %%
# Set up data
data_init = readFiles("expressions_example.csv", "copynumber_example.csv", "groundtruth_example.csv")
data_filt = geneDataFilter(data_init)
data_clean = cleanData(data_filt)
data_norm = normalizeData(data_clean) # whenever data_norm is changed, data_clean is also, idk whats going on
data_norm.exp.index = data_clean.exp.index # Somewhere the indices got messed up
data_norm.copy.index = data_clean.copy.index # But data_norm.truth is fine it seems
data_norm.truth.index = data_norm.truth.index + 1 # Since this started indexing at 0

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


'''


TODO:

- learn survival regression
- make a minimal submission


'''
