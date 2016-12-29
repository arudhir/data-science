# %%

COCKS
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
from sklearn.metrics import confusion_matrix, roc_curve
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

data_init.exp.shape

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
forest_reg_exp = regressionForest(train_data, data_norm.truth, 100)
forest_clss_exp = classificationForest(train_data, data_norm.truth, 10)


rankFeatures(forest_reg_exp, train_data)

# %%
# cross validation, i.e., using the model we just trained
data_cv = data_norm.exp.sample(frac=0.25)
xy_cv = alignData(data_cv, data_norm.truth)

X_cv = xy_cv[0]
y_cv = xy_cv[1]["PROGRESSED"]

forest_clss_exp.predict_proba(X_cv)


forest_clss_exp.predict_proba(X_cv)


forest_clss_exp == classificationForest(train_data, data_norm.truth, 10)




# %% Confusion Matrix
y_pred = forest_clss_exp.predict(X_cv)
# y_cv is what it "should" be

confusion_matrix(y_cv, y_pred)

confusion_matrix_arr = [confusion_matrix(y_cv, classificationForest(train_data, data_norm.truth, 10).predict(X_cv)) for x in range(20)]

confusion_matrix_arr = [confusion_matrix(y_cv, forest_clss_exp.predict(X_cv)) for x in range(20)]


true_neg = [confusion_matrix_arr[i][0][0] for i in range(20)]
false_neg = [confusion_matrix_arr[i][1][0] for i in range(20)]
true_pos = [confusion_matrix_arr[i][1][1] for i in range(20)]
false_pos = [confusion_matrix_arr[i][0][1] for i in range(20)]

sp.stats.describe(true_neg)
sp.stats.describe(false_neg)
sp.stats.describe(true_pos)
sp.stats.describe(false_)


for i in range(len(confusion_matrix_arr)):
    print(np.ndarray.view(confusion_matrix_arr[i]))
    print("\n")


plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
