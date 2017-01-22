import os
os.chdir("/Users/arudhir/Desktop/datascience/gen/data-science")
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer
from contextlib import suppress

class Data:
    '''
    This class instantiates Data objects that contain all the data, as well as various methods involved in pre-processing the data.
    '''

    def __init__(self, exp, cop, mut, labels):
        self.exp = exp
        self.cop = cop
        self.mut = mut
        self.labels = labels

    @classmethod
    def initFromFile(cls, path1, path2, path3, path4):
        '''
        Constructs Data object straight from .csv files.
        '''
        exp = pd.read_csv(path1)
        cop = pd.read_csv(path2)
        mut = pd.read_csv(path3)
        labels = pd.read_csv(path4)
        return cls(exp, cop, mut, labels)

    def addSuffix(self):
        '''
        Add suffixes to the columns in exp, cop, and mut, in order to distinguish them from each other.

        Note: Creates copy of dataframe. I don't think there is a "copy=False" parameter for add_suffix()
        '''
        self.exp = self.exp.add_suffix("_exp")
        self.cop = self.cop.add_suffix("_cop")
        self.mut = self.mut.add_suffix("_mut")
        return self

    def processLabels(self):
        '''
        Adds a new label called "PROGRESSED" indicating if a patient has progressed or not, as well as eliminates patients who have NAN in both the "TO" and "TP" columns.

        Warning: Does not create a copy of labels - it will modify the original.

        TODO: Fix the warning:
                SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame
        '''
        with suppress(ValueError):
            l = self.labels
            l.insert(len(l.columns), "PROGRESSED", 0)
            nan_indices = l["TO"].index[l["TO"].apply(np.isnan)]
            l["PROGRESSED"] = ~l["TP"].isnull()
            l["PROGRESSED"].ix[nan_indices] = np.nan
            self.labels = l
            return self
        return self

    def deleteProgressed(self):
        '''
        Convenience function to delete the "PROGRESSED" label from labels for testing.
        '''
        try:
            del self.labels["PROGRESSED"]
        except KeyError:
            pass

    def storeRawData(self, filename):
        '''
        Pickles the raw csv data so that we don't have to waste time preprocessing data more than once.
        '''
        joblib.dump(self, filename + ".pkl")

    def processData(self):
        '''
        Processes data.
        '''
        return self.addSuffix().processLabels().imputeData().labelsNoNan()

    def imputeData(self, mv="NaN", strat="mean", ax=0):
        '''
        Imputes the NaN values for all features, transforms the dataframe, and makes sure the columns are what they ought to be.
        Note: Imputer creates a copy internally.
        Args:
            mv (str): The type of feature Imputer finds. Default, NaN values
            strat (str): Strategy for Imputer. Default, mean
            ax: Axis for Imputer. Default, columns
        Return:

        '''
        imp = Imputer(missing_values=mv, strategy=strat, axis=ax)
        for k, v in self.__dict__.items():
            if k is not 'labels':
                col_name = pd.Series(v.columns.values)
                v = pd.DataFrame(imp.fit_transform(v))
                v = v.rename(columns=col_name)
                self.__dict__[k] = v
        return self

    def labelsNoNan(self):
        '''
        Get rid of patients who have NaN in "TO" and "TP"
        '''
        labels_nonan = self.labels.copy()
        nan_indices = self.labels["PROGRESSED"].loc[np.isnan(self.labels["PROGRESSED"])].index
        mask = labels_nonan["PROGRESSED"].index.isin(nan_indices)
        labels_nonan = labels_nonan[~mask]
        self.labels = labels_nonan
        return self

    def getXY(self):
        '''
        Get the X and y values for the actual model creation part.

        Note: np.ravel(ret2) is for creating an acceptable input for sklearn functions, which expect a 1D array for y, not a (306,) array (2D)
        '''
        # Concatenate all the X features`
        tempx = pd.concat([self.exp, pd.concat([self.cop, self.mut], axis=1)], axis=1)
        tempy = self.labels["PROGRESSED"]
        # Make sure the same samples exist in both X and y
        index = (tempx.index & tempy.index)
        ret1 = tempx.loc[index, :]
        ret2 = np.ravel(tempy.loc[index])

        return(ret1, ret2)

working_directory = "/Users/arudhir/Desktop/datascience/gen/data-science"
os.chdir(working_directory)
os.getcwd()
data = Data.initFromFile("expressions_example.csv", "copynumber_example.csv", "mutations_example.csv", "groundtruth_example.csv")

# Testing purposes only
exp = data.exp
cop = data.cop
mut = data.mut
labels = data.labels
d2 = Data(exp, cop, mut, labels)

d2.processData()
X, y = d2.getXY()
