# %%
import math
import numpy as np
import pandas as pd
import scipy as sp
import rpy2.robjects as robjects
import warnings
import sklearn
import matplotlib.pyplot as plt
from RFpyhelper import *
from rpy2.robjects.vectors import DataFrame
from rpy2.robjects.packages import importr, data
from rpy2.robjects import pandas2ri
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from lifelines import KaplanMeierFitter
from lifelines.utils import datetimes_to_durations, survival_table_from_events
from lifelines import AalenAdditiveFitter, CoxPHFitter
from scipy.interpolate import CubicSpline
from IPython.display import display, HTML
from collections import namedtuple
%matplotlib inline
pandas2ri.activate()
biocinstaller = importr("BiocInstaller")
genefilter = importr("genefilter")
warnings.filterwarnings('ignore')

# %%
# Set up data
data_init = readFiles("expressions_example.csv", "copynumber_example.csv", "groundtruth_example.csv")
data_filt = geneDataFilter(data_init)
data_clean = cleanData(data_filt)
data_norm = normalizeData(data_clean)
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

# %% Kaplan-Meier Survival Analysis
kmf = KaplanMeierFitter()

T = data_norm.truth["TO"]
E = data_norm.truth["PROGRESSED"]

kmf.fit(T, E) # Not sure if we should use the original/train/or CV data
kmf.survival_function_
kmf.median_
kmf.plot()
