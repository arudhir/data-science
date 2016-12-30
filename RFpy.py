# %%
import math
import numpy as np
import pandas as pd
import scipy as sp
import rpy2.robjects as robjects
import warnings
import sklearn
import matplotlib.pyplot as plt
import lifelines
from RFpyhelper import *
from rpy2.robjects.vectors import DataFrame
from rpy2.robjects.packages import importr, data
from rpy2.robjects import pandas2ri
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.manifold import Isomap
from sklearn.cluster.bicluster import SpectralCoclustering
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

rankFeatures(forest_reg_exp, train_data)

# %%
# cross validation, i.e., using the model we just trained
data_cv = data_norm.exp.sample(frac=0.25)
xy_cv = alignData(data_cv, data_norm.truth)

X_cv = xy_cv[0]
y_cv = xy_cv[1]["PROGRESSED"]

# %% Confusion Matrix
confusion_matrices = confusionMatrixStatistics(train_data, data_norm.truth, 10, xy_cv, 10)[0]
confusion_results = confusionMatrixStatistics(train_data, data_norm.truth, 10, xy_cv, 10)[1]
confusion_statistics = confusionMatrixStatistics(train_data, data_norm.truth, 10, xy_cv, 10)[2]


# %% ROC AUC score
# TODO: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
sklearn.metrics.roc_auc_score(y_cv, classificationForest(train_data, data_norm.truth, 10).predict(X_cv))
sklearn.metrics.roc_curve(y_cv, classificationForest(train_data, data_norm.truth, 10).predict(X_cv))

# %% Hazard and Survival Functions
# https://lifelines.readthedocs.io/en/latest/
orig_truth_clean = cleanData(data_filt).truth # Non-normalized data just bc idk what normalizing does to KM graphs
T = orig_truth_clean["TP"]
C = orig_truth_clean["PROGRESSED"]

# Survival Function
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

# %% Biclustering to get features
# experimental
clusters = sklearn.cluster.bicluster.SpectralCoclustering(n_clusters = 50)
fitted_clusters = clusters.fit(data_norm.copy)
pList = pd.DataFrame.as_matrix(progressedList(data_norm.truth["PROGRESSED"]))
biclusterCommon(fitted_clusters, pList)[40]
fitted_clusters.get_indices(40)[1]


'''


TODO:

survival analysis
=========================
learn about survival analysis
consider which model should be used for the function

conceptual
==========================
figure out how to integrate different pieces of data

'''
