import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import metrics

from mpl_toolkits import mplot3d

sns.set()


def argmedian(arr):
    """return index of median in an array"""
    return np.argpartition(arr, len(arr) // 2)[len(arr) // 2]


def plot_cmatrix(ytrue,
                 ypred,
                 cmapper,
                 title='',
                 ):

    c1 = np.array(metrics.confusion_matrix(ytrue, ypred))
    print(c1)

    pepcolor = np.unique(ytrue)

    fig, ax = plt.subplots()

    plt.suptitle('\n'+title)

    ax = sns.heatmap(c1/np.sum(c1, axis=1).reshape(-1, 1),
                     fmt='.1%',
                     annot=True,
                     cmap='Blues',
                     xticklabels=cmapper.inverse_transform(
                         np.arange(len(pepcolor))),
                     yticklabels=cmapper.inverse_transform(
                         np.arange(len(pepcolor)))
                     )
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.tick_params(labelsize=12)

    plt.show()

    return fig


def plot_bar(X,
             y,
             cmapper,
             title='',
             ):
    """
    X: 2-d array of xy-coordinates
    y: 1-d target array of numbers
    cmapper: labelencoder we defined to convert category into numbers
    """

    pepcolor = np.unique(y)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 10))
    fig.subplots_adjust(top=0.85)
    plt.suptitle('\n'+title, fontsize=24)

    plt.scatter(*X.T,
                c=y,
                cmap='Set1',
                alpha=1.0
                )

    ax.tick_params(labelsize=14)

    # you need a mapper object to plot cbar
    cbar = plt.colorbar(boundaries=np.arange(len(pepcolor)+1)-0.5)
    cbar.set_ticks(np.arange(len(pepcolor)))
    cbar.set_ticklabels(cmapper.inverse_transform(np.arange(len(pepcolor))))
    cbar.ax.tick_params(labelsize=14)

    plt.show()

    return fig


def plot_bar3d(X,
               y,
               cmapper,
               title='',
               elev=30,
               azim=30,
               ):
    """
    X: 3-d array of xy-coordinates
    y: 1-d target array of numbers
    cmapper: labelencoder we defined to convert category into numbers
    """

    pepcolor = np.unique(y)

    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection='3d')
    fig.subplots_adjust(top=0.85)
    plt.suptitle('\n'+title, fontsize=24)

    obj = ax.scatter(X[:, 0],
                     X[:, 1],
                     X[:, 2],
                     c=y,
                     cmap='Set1',
                     alpha=1.0
                     )

    ax.view_init(elev=elev, azim=azim)

    ax.tick_params(labelsize=14)

    # you need a mapper object to plot cbar
    cbar = fig.colorbar(obj, boundaries=np.arange(len(pepcolor)+1)-0.5)
    cbar.set_ticks(np.arange(len(pepcolor)))
    cbar.set_ticklabels(cmapper.inverse_transform(np.arange(len(pepcolor))))
    cbar.ax.tick_params(labelsize=14)

    plt.show()

    return fig


class FREQ_THRES(BaseEstimator, TransformerMixin):
    def __init__(self, freq_thres):
        self.freq_thres = freq_thres

    def fit(self, X, y=None):
        if self.freq_thres < 1 and self.freq_thres > 0:
            thres = len(X) * self.freq_thres
            X = X[X.columns[(X != 0).sum() >= thres]]

        self.columns = X.columns

        return self

    def transform(self, X, y=None):
        if self.freq_thres < 1 and self.freq_thres > 0:
            X = X[self.columns]

        return X


class LOG(BaseEstimator, TransformerMixin):
    def __init__(self, log):
        self.log = log

    def fit(self, X, y=None):
        if self.log:
            X = np.log(X + 1)
        return self

    def transform(self, X, y=None):
        if self.log:
            X = np.log(X + 1)
        return X


class REDUCE:
    def __init__(self, inputmd, inputml, output1):
        self.inputmd = inputmd
        self.inputml = inputml
        self.output1 = output1

    def read(self):
        self.df1 = pd.read_csv(self.inputmd,
                               index_col=0,
                               sep='\t',
                               )
        self.fname = self.df1.iloc[0, 0]

        print(self.fname)
        print('self.df1.shape', self.df1.shape)

    def select_pat(self, n, rseed=42):
        """randomly select n patients"""
        rng = np.random.RandomState(rseed)
    # pick four points out of 300 data points
        i = rng.permutation(len(self.df1))[:n]
        self.df1 = self.df1.iloc[i]
        print('self.df1.shape', self.df1.shape)

    def select_pep(self):
        """select sequenced peptides"""

        dfseq = pd.read_excel(self.inputml, index_col=0)

        # select sequenced peptide from ml1
        pepseq = dfseq[dfseq['Sequence'].notnull()].index.tolist()
        pepseq = np.sort(['x999'+str(i)[1:]for i in pepseq])
        pepseq = np.concatenate((['Krankheit'], pepseq))
#         print (pepseq)

        # filter only seq. peptides from raw
        self.df1 = self.df1[self.df1.columns[self.df1.columns.isin(pepseq)]]

        print('self.df1.shape', self.df1.shape)

    def select_disease(self, disease):
        """select patients with certain Krankheit"""
        self.df1 = self.df1[self.df1['Krankheit'].isin(disease)]
        print('self.df1.shape', self.df1.shape)

    def export_txt(self, suffix=''):
        self.df1.to_csv(self.output1+self.fname+suffix+'.csv')
