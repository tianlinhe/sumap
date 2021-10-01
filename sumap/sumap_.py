import pandas as pd
import numpy as np
from itertools import islice
import pickle

import umap
from sklearn.experimental import enable_halving_search_cv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV,\
    HalvingGridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.svm import SVC

import sumap.helpers as helpers


class SUMAP:
    def __init__(self,
                 gridsearch='half',
                 n_components=3,
                 classifier='svm',
                 cv=5,
                 min_resources='smallest',
                 factor=3,
                 score_func=metrics.f1_score,
                 average='weighted',
                 random_state=None,
                 n_jobs=-1,
                 **kwargs,
                 ):
        """Define a UMAP-SVM pipeline with tunable UMAP parameters
        Parameters
        ----------
        gridsearch: string
            'half': use HalvingGridSearchCV
            'full': use GridSearchCV
        n_components: int
            umap hyperparameters
        classifier: any sklearn compatible classifier
            default is SVC
        cv: int
            CV fold of grid search
        min_resources: string
            param of HalvingGridSearchCV
        factor: int
            param of HalvingGridSearchCV
        score_func: any sklearn.metrics score
        average: param of sklearn.metrics.f1_score
        random_state: int
            random state of umap and HalvingGridSearch and classifier
            if fixed, return reproducible results for the same input
        **kwargs: one or multiple lists with keyword,
            prerix by pipeline component + __
            e.g. n_neighbors is a component of umap
            -> umap__n_neighbors=[5, 15]

        Produce
        -------
        self.clf_pipeline, pipeline of
            data transform -> umap -> classifier
        """

        freq_thres = helpers.FREQ_THRES(freq_thres=0)

        log = helpers.LOG(log=False)

        if classifier == 'svm':

            classifier = SVC(random_state=random_state)

        mapper = umap.UMAP(random_state=random_state,
                           n_components=n_components)

        scorer = metrics.make_scorer(score_func=score_func,
                                     average=average,
                                     )

        pipeline = Pipeline([('freq', freq_thres),
                             ('log', log),
                             ('umap', mapper),
                             ('classifier', classifier),
                             ])

        params_grid_pipeline = {**kwargs}

        if gridsearch == 'full':
            self.clf_pipeline = GridSearchCV(pipeline,
                                             params_grid_pipeline,
                                             cv=cv,
                                             scoring=scorer,
                                             n_jobs=n_jobs,
                                             )
        elif gridsearch == 'half':
            self.clf_pipeline = \
                HalvingGridSearchCV(pipeline,
                                    params_grid_pipeline,
                                    cv=cv,
                                    scoring=scorer,
                                    n_jobs=n_jobs,
                                    min_resources=min_resources,
                                    factor=factor,
                                    random_state=random_state
                                    )

    def _load_data(self, X, y):
        """Load X and y, turn y to numericals
        X and y must NOT contains NaN

        Produce
        -------
        self.Xtrain=X
        self.ytrain=y
        """
        self.Xtrain = X

        # y if categories, needs to be transformed to number before fit
        self.le = LabelEncoder()
        self.ytrain = self.le.fit_transform(y)

    def fit(self,
            X,
            y):
        """Fit self.clf_pipeline
        Return
        ------
        self: for more "incremental" actions
            e.g. fit(X, y).transform(Xtest)
        """

        self._load_data(X, y)

        self.clf_pipeline.fit(self.Xtrain, self.ytrain)

        return self

    def transform(self,
                  X=None):
        """Embed X to lower dimension after fitting pipeline,
        run everything in pipeline until classifier
        Note
        ----
        Transform training data if X if not given

        Return
        ------
        numpy array with shape (len(X), n_components)
        """

        # self.clf_pipeline.best_estimator_ is an ordered dict
        # run everything in pipeline until classifier
        pipeline_transform = self.clf_pipeline.best_estimator_[:-1]

        if X is None:
            Xtransform_ = pipeline_transform.transform(self.Xtrain)
        else:
            Xtransform_ = pipeline_transform.transform(X)

        return Xtransform_

    def _predict(self,
                 X=None):
        """Predict label of X
        Note
        ----
        Transform training data if X if not given
        Return
        ------
        1-d numpy array with length len(X) as numericals
        """

        if X is None:
            ypred = self.clf_pipeline.predict(self.Xtrain)
        else:
            ypred = self.clf_pipeline.predict(X)

        return ypred

    def predict(self,
                X=None):
        """Predict label of X
        Note
        ----
        Transform training data if X if not given
        Return
        ------
        1-d numpy array with length len(X) as categoricals
        """

        ypred = self._predict(X)

        return self.le.inverse_transform(ypred)

    def score(self,
              X=None,
              y=None):
        """Calculate score of X and y from prediction
        Note
        ----
        If X and y not given, return training score
        Return
        ------
            score between 0 and 1
            score depends on score_func defined
        """
        if X is None and y is None:
            X = self.Xtrain
            y = self.ytrain
        elif X is not None and y is not None:
            if isinstance(y, pd.Series):
                y = self.le.transform(y)
        else:
            raise ValueError

        return self.clf_pipeline.score(X, y)

    def plot_cmatrix(self,
                     X=None,
                     y=None,
                     ):
        """Plot confusion matrix given X and y
        Note
        ----
        If X and y not given, use training data
        Return
        ------
        matplotlib figure object
        can be saved as obj.savefig('')
        """
        if X is None and y is None:
            y = self.ytrain
        elif X is not None and y is not None:
            if isinstance(y, pd.Series):
                y = self.le.transform(y)
        else:
            raise ValueError

        ypred = self._predict(X)

        return helpers.plot_cmatrix(y,
                                    ypred,
                                    cmapper=self.le,
                                    )

    def plot_embeddings(self,
                        X=None,
                        y=None,
                        elev=30,
                        azim=30,
                        ):
        """Plot low-dimensional embeddings using X, color with y
        Parameter
        ---------
        elev: float
            elev of a 3d plot
        azim: float
            azim of a 3d plot
        Note
        ----
        If X and y not given, use training data

        Return
        ------
        matplotlib figure object
        can be saved as obj.savefig('')
        """

        # get embeddings from the fitted pipeline
        Xtransform_ = self.transform(X)

        if X is None and y is None:
            cmapper_ = self.le
            y = self.ytrain
        elif X is not None and y is not None:
            cmapper_ = self.le
            if isinstance(y, pd.Series):
                cmapper_ = LabelEncoder()
                y = cmapper_.fit_transform(y)
        else:
            raise ValueError

        # plot 2d if Xtransform has 2 columns
        if Xtransform_.shape[1] == 2:
            return helpers.plot_bar(Xtransform_,
                                    y,
                                    cmapper=cmapper_,
                                    title='N = {n}'.format(n=len(y)),
                                    )

        # plot 3d if Xtransform has 3 columns
        elif Xtransform_.shape[1] == 3:
            return helpers.plot_bar3d(Xtransform_,
                                      y,
                                      cmapper=cmapper_,
                                      title='N = {n}'.format(n=len(y)),
                                      elev=elev,
                                      azim=azim,
                                      )


class SUMAP_nestedCV(SUMAP):

    def __init__(self,
                 n_splits_outer=5,
                 gridsearch='half',
                 n_components=3,
                 classifier='svm',
                 cv=5,
                 min_resources='smallest',
                 factor=3,
                 score_func=metrics.f1_score,
                 average='weighted',
                 random_state=None,
                 n_jobs=-1,
                 **kwargs,
                 ):
        """Define a UMAP-SVM pipeline with tunable UMAP parameters
        Parameters
        ----------
        n_splits_outer: int
            outer CV fold for nestedCV
        all other params inherited from SUMP
        Produce
        ------
        self.clf_pipeline, pipeline of
            data transform -> umap -> classifier
        self.outer_cv: sklearn StratifiedKfold obj
        """

        self.outer_cv = \
            StratifiedKFold(n_splits=n_splits_outer,
                            shuffle=True,  # each fold is independent
                            random_state=random_state)

        super().__init__(gridsearch,
                         n_components,
                         classifier,
                         cv,
                         min_resources,
                         factor,
                         score_func,
                         average,
                         random_state,
                         n_jobs,
                         **kwargs,
                         )

    def fit(self,
            X,
            y,
            estimator=None,
            ):
        """Nested CV to fit self.clf_pipeline
        for OuterCV: Samples were split to train and test
            InnerCV: Model is developed in train
        Model scores recordedin test

        Parameters
        ----------
        - estimator: string, specify the model to return
            If 'best' return the one with highest test score
            If 'random' return a randomly chosen model from outer CV
            If 'average' return the one with median performace
            else return the last one (default)

        Produce
        -------
        self.outer_testscores: list with len(n_splits_outer)
            model scores recorder in outerCV
        self.Xtrain(updated) that generates the test score
        self.ytrain(updated) that generates the test score
        self.clf_pipeline fitted by updated self.Xtrain, self.ytrain

        Return
        ------
        self: for more "incremental" actions
            e.g. fit(X, y).transform(Xtest)
        """

        self._load_data(X, y)

        self.outer_testscores = []
        self.outer_trainscores = []
        self.outer_pipelines = []

        n_outer = 0
        for train_id, test_id in self.outer_cv.split(self.Xtrain, self.ytrain):
            Xtrain, Xtest = \
                self.Xtrain.iloc[train_id], self.Xtrain.iloc[test_id]
            ytrain, ytest = self.ytrain[train_id], self.ytrain[test_id]

            self.clf_pipeline.fit(Xtrain, ytrain)

            self.outer_pipelines.append(pickle.dumps(self.clf_pipeline))

            # measure the model performance (chosen by inner cv) on outer cv
            Xtest_score = self.clf_pipeline.score(Xtest, ytest)
            Xtrain_score = self.clf_pipeline.score(Xtrain, ytrain)

            self.outer_testscores.append(Xtest_score)
            self.outer_trainscores.append(Xtrain_score)

            print('n_outer =', n_outer)
            print('train score =', Xtrain_score)
            print('test score =', Xtest_score)

            n_outer += 1

        print('Scores of {n} models: {mean} +/- {std}'
              .format(n=n_outer,
                      mean=np.mean(self.outer_testscores),
                      std=np.std(self.outer_testscores)
                      )
              )

        if estimator == 'best':
            chosen_split = np.argmax(self.outer_testscores)

        elif estimator == 'random':
            chosen_split = np.random.choice(n_outer)

        elif estimator == 'average':
            chosen_split = helpers.argmedian(self.outer_testscores)

        try:
            print('chosen_split', chosen_split)
            train_id, test_id = next(islice(self.outer_cv.split(self.Xtrain,
                                                                self.ytrain),
                                            chosen_split,
                                            None
                                            )
                                     )
            self.Xtest, self.ytest = \
                self.Xtrain.iloc[test_id], self.ytrain[test_id]

            self.Xtrain, self.ytrain = \
                self.Xtrain.iloc[train_id], self.ytrain[train_id]

            # chose the model, pickle to avoid refit
            self.clf_pipeline = pickle.loads(
                self.outer_pipelines[chosen_split])

        except NameError:
            self.Xtrain, self.ytrain = Xtrain, ytrain
            self.Xtest, self.ytest = Xtest, ytest

        return self
