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
from sklearn.ensemble import RandomForestClassifier
from scipy.interpolate import CubicSpline
from IPython.display import display, HTML
from collections import namedtuple
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

# Sample 50% of the data for the test set
data_train = data_norm.exp.sample(frac=0.5)

xy_train = alignData(data_train, data_norm.truth)

# %%
# TIME TO DO SOMETHING

clf = RandomForestClassifier(n_estimators=100)
X = xy_train[0]
y = xy_train[1]["PROGRESSED"]

rf = clf.fit(X, y)

data_cv = data_norm.exp.sample(frac=0.25)
xy_cv = alignData(data_cv, data_norm.truth)

rf.predict_proba(test_xy[0])
