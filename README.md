# SUMAP: Supervised UMAP

`sumap` makes it easy to tune the parameters of UMAP (Uniform Manifold Approximation and Projection) for enhancement of embedding of high-dimensional data. 

## Installation

`sumap` can be installed by running the following command line code:

```
pip install sumap
```

## General Usage
By creating a `sumap.SUMAP()` instance, one can set up the search of UMAP hyperparameters in a grid search by prefixing the hyperparameters with `umap__`.

By default, the optimal UMAP hyperparamters is selected by a SVC classifier.

```
import sumap

# create sumap instance
mypipeline = sumap.SUMAP(
    umap__n_neigbors=[5, 10] # list of n_neighbors to search
    umap__min_dist=[0, 0.5] # list of min_dist to search
)
```

Then we can fit the instance with training data:

```
# fit sumap instance with training data
mypipeline.fit(Xtrain, # Pandas dataframe
               ytrain # Pandas series
               )
```

The fitted pipeline is stored as a `Pipeline` object which can be accessed by `mypipeline.clf_pipeline`.

Alternatively, `mypipeline` has some attributes as function that allows one to transform data into lower dimensional, and predict the label of data etc.

```
# transform data to n_components dimension
mypipeline.transform(Xtest)

# predict label
mypipeline.predict(Xtest)

# score the predicted labels, if true labels were given
mypipline.score(Xtest, ytest)
```

## Plotting utilities
In addition, one can access more attributes for easy plotting of the results:

```
# plot confusion matrix of the classification, if true labels were given
mypipeline.plot_cmatrix(Xtest, ytest)

# plot the optimal umap embeddings and color the labels if given
mypipeline.plot_embeddings(Xtest, ytest)
```

## Links:
* [SUMAP on GitHub](https://github.com/tianlinhe/sumap)
* [UMAP on GitHub](https://github.com/lmcinnes/umap)
