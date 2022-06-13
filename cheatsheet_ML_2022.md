# Machine Learning Algorithms
## Clustering Algorithms
### k-Means
- assume k clusters
- loss function: for each point determine the minimum distance to all clusters and sum these distances up across all points. Minimize the sum of all distances wrt to the position of the clusters.
- regularization parameters: could be the distance between clusters or a penalty term for the number of clusters or calculate the distance^k.
- start values: one in the center of gravity, then along the principal components.

in https://scikit-learn.org/stable/modules/clustering.html#k-means
- start with clusters
- minimize in-cluster distances
- Voronoi-Diagrams
- assumes convex shaped clusters
- PCA can help

- starting point: select k clusters
- iterate:
    - assign to cluster
    - calculate new centroid based on the members of the cluster
- stop until the cluster centroids barely move

- "k-means++" initialization
- Mini-batch 

excellent description:
https://towardsdatascience.com/explaining-k-means-clustering-5298dc47bad6
- k-Means in scikit learn has a parameter n_int= 20 -> select 20 random clusters
- number of iterations

- all features have to have the same scale 

### dimensionality reduction: PCA, t-SNE
- PCA: Eigenvalues of data
- t-SNE: T-distributed stochastic neighbor embedding
- [FeatureAgglomoration](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html#sklearn.cluster.FeatureAgglomeration)
- PCA removes _linear_ correlations across features (https://scikit-learn.org/stable/modules/preprocessing.html)

### t-SNE: t-distributed stochastic neighbor embedding
- [Original paper](https://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
- A tool to visualize high-dimensional data.
- random walk version of t-SNE
- t-SNE reduces dimensionality mainly based on local structure of the data
- not obvious how t-SNE performs for dimension d larger than 3, it is optimized for d=2 or 3
- perform t-SNE on output of an autoencoder is likely to improve the quality of the visualizations.
- many dimensionality reduction techniques have convex cost functions, but not t-SNE.



### [DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
Areas of high density surrounded by areas of low density. 

Keywords: core_sample, eps and min_sample parameters

a local definition of a cluster from core samples and chaining them together

A core_samples is defined as a subsample of the dataset with each core_sample having at least min_sample other core_samples within a distance of eps.
Thus core_samples are within a dense area of the space in which the dataset lives.
A cluster is a set of core_samples that have neighbors as core_samples, who again have core_samples as neighbors. Samples that are not core_samples anymore, because of less than min_samples within a distance eps define the edge of a cluster.

#### Advantages over k-Means
does not require the clusters to be convex and isotropic.

### hierarchical clustering

## [data preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- Standarization as a common requirement for ML estimators: approximately normal distributed
- subtract mean and normalize by the variance
- train_test_split()
- make_pipeline(StandardScaler(), LogisticRegression())
- RobustScaler(): when there are many outliers
- Mapping to a Gaussian distribution (interesting. How to map it back?)
- Encoding categorical features: preprocessing.OrdinalEncoder(encoded_missing_value= -1)

### Normalization
Normalization is the process of scaling individual samples to have unit norm.

### Encoding categorical features
- OrdinalEncoder()
- OneHotEncoder()
    - enc.categories_ - this illustrates the vector
    - enc.transform()
    - infrequent categories: group them into one - Arguments of OneHotEncoder()
- Discretization: quantization or binning


### [Generating polynomial features](https://scikit-learn.org/stable/modules/preprocessing.html#generating-polynomial-features)

### Custom Transformers

- ```from sklearn.preprocessing import FunctionTransformer```
- something like an DataFrame::apply() with a lambda function

## [Imputation](https://scikit-learn.org/stable/modules/impute.html)
- most classifiers or regressors do not accept missing values
- imputation is the process of filling in variables with values calculated under certain assumptions
- [marking imputed values](https://scikit-learn.org/stable/modules/impute.html#marking-imputed-values)
- [Univariate vs. multivariate imputation](https://scikit-learn.org/stable/modules/impute.html#univariate-vs-multivariate-imputation)

## Decision Trees
- a supervised learning technique
- for classification and regression
- for tabular data, but also works for faces
- a model that predicts the value of a target variable by learning
- piecewise constant approximation
- approximating a sin curve with piecewise constant parts (if-else conditions)

### Advantages
- trees can be visualized, easy to understand
- numerical and categorical data
- low data preparation requirements -> low threshold to run it.

### Disadvantages
- prone to overfitting, especially for big amounts of features
- depending on details the tree can have very different shapes
    - addressed via ensemble methods
- needs a balanced dataset

### Classification
- class DecisionTreeClassifier: multi-class
- methods:
    - fit()
    - predict()
    - predict_proba()
    - plot_tree()
    - export_graphviz()

### Multi-output problems
a multi-output problem is a supervised learning problem with n-dimensional output with n>1. I.e. the output has dimension (n_samples, n_outputs).
If there are no correlations between the outputs and the inputs leading to these outputs, one can train individual models for each 1-dimensional output and combine them in the end.

If they are correlated a better method is to train one model capable of predicting simultaneously all n outputs.

1. lower training time required to built only one estimator
2. generalization power is often increased.

### Regressor
- class DecisionTreeRegressor

### Practical Tips
- works well with PCA or LDA
- start with max_depth=3 as initial depth, look at the data, then increase the depth

## [Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
In order to increase robustness and generalizability individual estimators can be combined to form a more powerful ensemble.

There are two types of ensemble methods:
1. averaging: The variance of the ensemble is reduced compared to each individual estimator's prediction (bagging methods, random forests)
2. in boosting methods, base estimators are built sequentially with the goal to reduce the bias of the combined estimator. The motivation is to combine several weak models to produce a powerful ensemble (AdaBoost, Gradient Tree Boosting)

### Bagging meta-estimator
- several instances of a black-box estimator on random subsets of the original training set and then aggregate their predictions to form a final prediction.
- reduce variance 
- avoid overfitting
- bagging methods work best with strong and complex models
- in contrast with boosting methods which usually work best with weak models (e.g. shallow deicsion trees)
- different bagging methods differ in the way they draw subsets of the training set.
- BaggingRegressor() and BaggingClassifier()

### Forests of randomized trees
sklearn.ensemble knows two averaging algorithms based on randomized decision trees using perturb-and-combine techniques:
- the RandomForest algorithm
- the Extra-Trees method
The prediction is the averaged prediction of the individual classifiers.
Reduce variance through introducing randomness. Individual decision trees exhibit high variance and tend to overfit. The injected randomness in forests yield decision trees with somewhat decoupled prediction errors. By taking the average of many such trees, certain errors cancel out.

#### Random Forests
- RandomForestClassifier, RandomForestRegressor
- two sources of randomness: random subsample draw and splitting
    - each tree in the ensemble is built from a sample drawn with replacement
    - when splitting each node during the construction of a tree, the best split is found either from all input features or a random subset of size max_features.
- Random forests achieve a reduced variance by combining diverse trees, sometimes at the cost of a slight increase in bias.
- look at most discriminative thresholds
#### Extremely randomized trees
... exist as well :)

#### Parameters
- n_estimators (the larger the better, but it takes longer to compute) and 
- max_features (regression: max_features=1.0 down to 0.3, classification: max_features="sqrt")
- results will stop getting significantly better beyond a critical number of trees.
- (...)

### Evaluation and Feature importance evaluation
- based on out-of-bag samples: (oob_score=True)
- the higher up in the tree, the more samples are affected
- expected fraction of the samples as an estimate of the relative importance of the features.
- feature_importances_

### AdaBoost
- 1995 by Freund and Schapire
- fit a sequence of weak learners on repeatedly modified versions of the data. (weak learners are models only weakly better than random guessing).
- A weighted majority of these trees is then the overall prediction.
- boosting iteration: calculate different weights to each training sample (initially 1/N)
- weights of misclassified samples are increased, those of rightly classified samples are decreased. Thereby over the course of several iterations the misclassified samples gain ever more weight and the algorithm has to focus on them.

#### Parameters
- n_estimators (number of weak learners, e.g. 100)
- max_depth: the depth of the base estimators

### Gradient Tree Boosting
- generalization of boosting to arbitrary differentiable loss functions.
(...)
### Histogram-Based Gradient Boosting
(...)

## [Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- ```sklearn.feature_selection```
### Removing features with low variance
- VarianceThreshold
- it requires the variance of one feature to go above a given threshold


### Univariate feature selection
- Univariate feature selection works by selecting the best features based on univariate statistical tests.
- scikit-learn exposes feature selection routines as objects that implement the transform method. (why not select-method?)
- SelectKBest(), SelectPercentile()
- Warning: do not use regression scoing functions with a classification problem and vice versa.

### Other methods
- Recursive feature elimination
- Feature selection using SelectFromModel
    - e.g. a Tree-based feature selection
- Sequential Feature Selection
- Feature selection as part of a pipeline

## [Semi-supervised learning](https://scikit-learn.org/stable/modules/semi_supervised.html)
- Some of the training data is not labeled. The semi-supervised estimators in sklearn.semi_supervised are able to make use of this additional unlabeled data to better capture the shape of the underlying data distribution and generalize better to new samples.
- Particularly suited in case of few labeled and much unlabeled data.
- unlabeled entries in Y get a dedicated identifier along with the labeled data when training the model with the fit method.

1. estimator trained in supervised way with labeled data
2. use the classifier above to train it further in unsupervised fashion

- SelfTrainingClassifier requires predict_proba()
- in each iteration the classifier predicts labels for the unlabeled samples and adds a subset of these labels to the labeled dataset.

### [Semi-supervised learning (Wikipedia)](https://en.wikipedia.org/wiki/Semi-supervised_learning#Assumptions)
- falls between supervised and unsupervised.
- special instance of weak supervision.
- acquisition of unlabeled data is relatively inexpensive
- transductive or inductive learning: The goal of transductive learning is to infer the correct labels for the given unlabeled data. The goal of inductive learning is to infer the correct mapping from input features to output targets.
- in practice, algorithms formally desinged for transduction or induction are often used interchangeably.

#### Assumptions
Semi-supervised learning applies at least one of the following assumptions
- continuity assumption: points that are close to each other are more likely to share a label. This is also a common assumption in supervised learning and gives rise to a simple decision boundary
- cluster assumption: data tend to form clusters and points in the same cluster are more likely to share a label. But data with the same label could spread across different clusters.
- manifold assumption: the data lie approximately on a manifold of much lower dimension than the input space. Learning can then proceed using distances and densities defined on the manifold.
E.g. human voice is controlled by a few vocal folds and facial expressions are controlled by a few muscles. In these cases distances and smoothness in the manifold of the problem are much reduced compared to the full space of all possible acoustic waves or images of faces.

#### Further Reading
[Efficient Non-Parametric Function Induction in Semi-Supervised Learning](http://nicolas.le-roux.name/publications/Delalleau05_ssl.pdf)

## [Probability Calibration](https://scikit-learn.org/stable/modules/calibration.html)
- predict class label with a probability
- not all estimators provide a probability
- "Well calibrated classifiers are probabilistic classifiers for which the output of the predict_proba() method can be directly interpreted as a confidence level.
For instance, a well calibrated (binary) classifier should classify the samples such that among the samples to which it gave a predict_proba value close to 0.8, approximately 80 % actually belong to the positive class." (statistical interpretation)
### Calibration Curves
- calibration curves (aka reliability diagrams) for a binary classifier
- CalibratedClassifierCV

#### Further Reading
- [Beyond Sigmoids](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-11/issue-2/Beyond-sigmoids--How-to-obtain-well-calibrated-probabilities-from/10.1214/17-EJS1338SI.full)
- [Predicting Good Probabilities With Supervised Learning](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)

## Neural network models (supervised)
- scikit-learn offers no GPU support
- any model estimation is sample-dependent (from [Neural Networks for regression and classification](https://github.com/joseDorronsoro/Neural-Networks-for-Regression-and-Classification/blob/master/notes_NN_regr_class_2021.pdf))
### Multi-Layer Perceptron (MLP)
- a supervised learning algorithm that learns a function f(R^m -> R^o) by training on a dataset, where m is the number of dimensions for input and o the number of output dimensions
- either classification or regression
- hidden layers
- building blocks: neurons and arrows make the output of the previous layer the input for the current layer
- weighted linear summation + biases followed by a non-linear activation function g(R -> R)
- MLPClassifier, MLPRegressor (implements predict_proba())

### Advantages
- learn non-linear functions/models
- learn models in real-time (online learning) using ```partial_fit```

### Disadvantages
- non-convex loss function with more than one local minimum
- sensitive to feature scaling

### Examples of activation functions
- sigmoid function $\frac{1}{(1 + e^{-x})}$
- tanh is s-shaped like the sigmoid function (-1 to 1)
- ReLU (Rectified Linear Unit) Activation function: $R(x) = max(0, x)$: non-saturating ReLU activation function, it shows improved training performance over tanh and sigmoid.
- leaky ReLU: $R(x) = max(a*x, x)$ with 0 < a << 1
- softmax function (last layer of a classifier): ratio of $\exp(i)/\sum_j(\exp(j)$ - for multiple classes

### Linear Regression
- no activation function in the output layer
- continuous values in the output layer 
- use square error as loss function to minimize the distance between $|y^2-y^{\hat}^2|$

### Classification
- Classifiers don't have a distance metric defined for the target classes, instead take the most probable label given the inputs x. Linear regression problems have such a distance metric and therefore the L2- or L1-distance can be used as a loss function.
- interesting link: [Neural Networks for regression and classification](https://github.com/joseDorronsoro/Neural-Networks-for-Regression-and-Classification/blob/master/notes_NN_regr_class_2021.pdf)
    - good probability estimates are useful
    - concrete labels used for targets do not matter much
    - model learning should thus be target-agnostic

### Examples of loss functions
- mean squared error (MSE)
- y = f(x) can be:
    - logistic loss function, mean squared error
    - SVMs, Random Forest, decision trees, gradient boosted, nearest neighbor


### Regularization
See [7 regularization for deep learning](summary_Neural_Networks.md#7-regularization-for-deep-learning)

### Complexity
For a training set of size n, split into several batches of size b, m features, k hidden layers, each containing h neurons and o output neurons. The time complexity of backpropagation is $O(b\cdot m\cdot h^k\cdot o\cdot i)$, where i is the number of iterations.

### Curse of dimensionality
High dimensional spaces become quickly extremely sparsely populated, where the data of interest lives in a low dimensional manifold. Distance measures as loss function become numerically challenging.

"As the dimensionality increases, the number of data points required for good performance of any machine learning algorithm increases exponentially. The reason is that, we would need more number of data points for any given combination of features, for any machine learning model to be valid."
from: [curse of dimensionality](https://towardsdatascience.com/curse-of-dimensionality-a-curse-to-machine-learning-c122ee33bfeb)

On distance: difference between min and max distance goes towards 0 for increasing dimensions. 


## Projects related to scikit-learn
- many projects listed
- shortlist:
    - seaborn: matplotlib based library for attractive plotting
    - Keras: high-level API for TensorFlow with a scikit-learn inspired API
    - dtreeviz: a python library for decision tree visualization and model interpretation
    - sklearn-pandas: bridge for scikit-learn pipelines and pandas data frame with dedicated transformers
    - [scikit-lego](https://github.com/koaning/scikit-lego): custom transformers, metrics and models (industry focus)
    - [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn): various methods to under- and over-sample datasets

### Recognizing Overfitting
- decision boundary plot has strong curvatures
- learning curve of training dataset is small, learning curve of test dataset is big
- [high variance](https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html#varying-regularization-in-multi-layer-perceptron)

## Neural Networks
see also [summary_Neural_Networks.md](summary_Neural_Networks.md)


## [Model selection and evaluation](https://scikit-learn.org/stable/model_selection.html)
### Cross Validation: evaluating estimator performance
- machine learning "experiment"
- cross validation workflow in model training: the best (hyper-)parameters can be determined by grid search techniques
- ```from sklearn.model_selection import train_test_split```
- 3 datasets:
    - training set
    - validation set to find hyper parameters
    - final evaluation on test set
- downside: much reduced data for training
- solution: k-fold cross validation (CV)
    - split the data into k parts
    - train on k-1 parts
    - optimize parameter on the k-th part
- ```from sklearn.model_selection import cross_val_score```
- ```from sklearn.model_selection import cross_validate```
    - allows specifying multiple metrics for evaluation
    - returns a dict containing fit-times, score-times (and optionally training scores as well as fitted estimators) in addition to the test score.
    - scoring parameter: precision_macro, recall_macro

### Obtaining predictions by cross-validation
### Cross validation iterators
Assumption: independent and identically distributed: assumption that all samples are drawn from the same underlying distribution and there is no hysteresis while generating from this distribution.

#### Alternatives:
- time-series aware cross-validation scheme
- group-wise cross-validation

### Cross validation iterators for grouped data
- several samples from the same patient form one group
- could be interesting for the OPEN diabetes project

### [Cross validation iterators for time series data]
Time series data is characterized by the correlation between observations that are near in time (autocorrelation). Classical CV techniques assume that the samples are independent and identically distributed, and would result in unreasonable correlation between training and testing instances.

```TimeSeriesSplit```

### Cross validation and model selection
### Permutation test score
Shuffle labels Y wrt the input X of left out data, thereby removing any dependency between the features and the labels.
The p-value is hte fraction of permutations for which the average cross-validation score obtained by the model is better than the cross-validation score obtained by the model using the original data.

A low p-value provides evidence that the dataset contains real dependency between features and labels and that the classifier manages to represent this dependency.

It is important to note that this test has been shown to produce low p-values even if there is only weak structure in the data because in the corresponding permutated datasets here is absolutely no structure.

### Tuning the hyper-parameters of an estimator
- hyper-parameters are parameters that are not optimized within the estimator. In scikit-learn they are passed as arguments to the constructor of the estimator classes.
 search the hyper-parameter space for the best CV scores.
- any parameter provided to the estimator's constructor may be optimized in this manner.
- ```estimator.get_params()```

#### Two generic approaches:
- GridSearchCV
- RandomizedSearchCV
- HalvingGridSearchCV
- HalvingRandomSearchCV

\[...\] much more detail is possible. Left out for now (May 22, 2022)

### Tips for parameter search
#### Specifying an objective metric
By default, parameter search uses the score function of the estimator to evaluate a parameter setting:
- For classification: ```sklearn.metrics.accuracy_score```
- for linear regression: ```sklearn.metrics.r2_score```

### Alternative to brute force parameter search
- ```linear_model.ElasticNetCV``` and other ```linear_model.*CV```
- Out of bag estimates

### Metrics and scoring: quantifying the quality of predictions
#### score examples

- f1-score: harmonic mean of precision and recall (two element classification)
- accuracy: on-diagonal elements (two element classification)

### classification metrics
- loss, score and utility functions to measure classification performance.
- precision_recall_curve: compute precision-recall pairs for different probability thresholds.
- roc_curve: compute receiver operating characteristic (ROC)
- confusion_matrix: generalization of the binary TP,FP,FN,TN-Matrix, comes with ConfusionMatrixDisplay
- f1_score: (see above)
- accuracy_score: (see above)
- top-k accuracy_score: generalization of accuracy_score: for images
- _beware of imbalanced datasets_
- classification_report: very call summary including precision, recall, F1, others

### Precision, recall and F-measures
- $F_beta$: a generalization of F1 (F1: beta=1), changing the relative weight of precision and recall
- value range: [0, 1]
- precision, recall, F1-score can be calculated per class

### ROC - Receiver Operating Characteristic
- from https://towardsdatascience.com/handling-imbalanced-datasets-in-machine-learning-7a0e84220f28
- assumption: we have a model that provides a probability that a certain point belongs to C: $P(C | x)$. Based on this probability function one can calculate a isoline where the probability is equal to a certain threshold T: $P(C | x) \geq T$. 
- good model: one does not have to sacrifice a lot in precision in order to get high recall.
- 

### Log Loss
- log loss, logistic regression loss or cross-entropy loss
- commonly used in (multinomial) logistic regression and neural networks 
- predict_proba()

## Glossary
- RANSAC: random sampling consensus - Zufallsstichprobe
- Bottom-Up- and Top-Down-Approaches: Top is where one is, Bottom is where many are.
- splines: piece-wise polynomials
- [scikit-learn Glossary](https://scikit-learn.org/stable/glossary.html#glossary)
- https://en.wikipedia.org/wiki/Duck_typing
- Overfitting: describes the training data well, but does not generalize well to independent test datasets
- Regression: 
    - _regress_ y back into the input feature vector x
    - there is the overloaded meaning: logistic regression (classification) and linear regression (to refer to both as in the formulation above) or: (linear) regression vs. classification (to distinguish the two)
- Model Evaluation: $R^2 = 1 - \frac{MSE}{Var(y)}$
- '[cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))' aka 'rotation estimation' or 'out-of-sample testing'
- RBF: radial basis function: $\sum \exp(-|W - X|)$ - difficult to optimize as it saturates to 0 for most x (Goodfellow, DL)
- Bagging: bootstrap aggregating
## Interesting AOB
- [bias-variance decomposition](https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html): in regression the mean squared error can be decomposed in terms of bias, variance and noise.
- Boolean features are Bernoulli random variables
- [Kernel methods to project data into alternate dimensional spaces](https://scikit-learn.org/stable/modules/semi_supervised.html#label-propagation)
- [ADAM optimizer](https://arxiv.org/pdf/1412.6980.pdf) by Kingma, Ba
- Spikes during training
    - twitter thread with Jeremy Howard, May 22, 2022
    - leads to many dead areas: large gradient, strong negative weights, ReLU goes negative, gradient 0, does not recover
    - solution: go back to an earlier checkpoint, skip some batches
    - is it the positive or negative side of the spike that leads to these negative weights?
    - see learner.activation_stats.color_dim
- MLPerf of MLcommons.org (referenced above)
- Distributed Shampoo: a scalable second order optimization method for deep learning
- wandb.ai: weights and biases
- iid aka i.i.d.: independent and identically distributed
- [epoch vs. iteration](https://stackoverflow.com/questions/4752626/epoch-vs-iteration-when-training-neural-networks): 
    - "in neural network terminology:
        - one epoch: one forward pass and one backward pass of all the training examples
        - batch size: number of training examples in one forward/backward pass
        - number of iterations: number of passes, each pass using [batch size] number of examples."
- Carsten K., May 2022: "egal welches Framework Du verwendest, die Netzwerkbeschreibung wird mit Keras gemacht. Das läuft bei mir alles unter Tensorflow"
- having an idea vs. popularizing it: https://medium.com/syncedreview/who-invented-backpropagation-hinton-says-he-didnt-but-his-work-made-it-popular-e0854504d6d1

## Book Recommendations
- Twitter feed (Chris Albon, Mai 22, 2022)
- "What are the best ML books to come out in the last three years? They can be code-focused, theory-focused, whatever, but they have to be books"
- Deep Learning with Python
- Machine Learning Design Patterns: Solutions to Common Challenges in Data Preparation, Model Building and MLOps (Google Cloud centric, Valliappa Lakshmanan, Sara Robinson & Michael Munn)
- Natural Language Processing with Transformers: Building Language Applications with Hugging Face
- Designing Machine Learning Systems: An Iterative Process for Production Ready Applications (Chip Huyen)
- Python: Data Science Handbook


## 2.7 Anomaly detection: Novelty and Outlier Detection
- https://scikit-learn.org/stable/modules/outlier_detection.html
- clean real data sets
- tails of a distribution or in the main distribution
- outlier: unknown, separate distribution compared to a known, expected distribution
- novelty detection vs. outlier detection: the former: whether a new entry belongs to an existing main or unpolluted distribution
- a distance measure: "far" from a main distribution
- anomaly detection:
    - outlier detection: unsupervised anomaly detection
    - novelty detection: semi-supervised anomaly detection


- outlier detection: do not form a cluster themselves, low density area
- novelty detection: could form a dense cluster, as long as they are in a low density area of the training data

- anomaly detection in scikit-learn:
- learn in an unsupervised learning fashion
- ```estimator.fit(X_train)```
- ```estimator.predict(X_test)```: inliers: 1, outliers: -1
- predict method: threshold on a scoring function (accessable via - ```estimator.score_samples(X_test)```) learned on the training data.
- decision function: positive or negative score for inliers or outliers

- neighbors.LocalOutlierFactor() has an outlier and a novelty detection mode. (see below)

- PCA is a very good first step for anomaly detection to minimize the curse of 


### my own idea of anomaly detection (unsupervised)
- take your preferred clustering algorithm
- describe each cluster in its own coordinate system. Perform PCA with a var (90 % or 95 %) criterium. 
- Apply a rms (default: three) RMS criterium to select inliers around these clusters. 
- Determine inliers of all clusters: points can also be inliers to more than one cluster. 
- All remaining points are then outliers.
- Merge clusters for which there is a significant fraction of overlap of common inliers
- tails of distributions

#### Future steps: 
- estimate the manifold in which the cluster lives
- use DBSCAN to get the high density points, which then form the manifold.
- estimate outliers relative to these manifolds 

### 2.7.1 Overview of Outlier Detection Methods
- IsolationForest
- [One-Class SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html) (classic or stochastic): unstable against outliers, uses libsvm, estimate the support of high dimensional distribution.
- LocalOutlierFactor

### [neighbors.LocalOutlierFactor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
- second description: https://scikit-learn.org/stable/modules/outlier_detection.html#local-outlier-factor
- novelty flag
- novelty detection: fit(X_train), then predict() on new test data 
- outlier detection: fit_predict(X) (contains already inliers and outliers)

### [IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
Return the anomaly score of each sample using the Isolation Forest Algorithm.

The IsolationForest algorithm isolates outliers by selecting a feature and determining a split between in- and outliers based on the min and max in this feature.
Recursive partitioning can be represented as a tree structure. The number of splittings required to isolate samples is equivalent to the path length from finale node to root node. This path length averaged over a forest of such trees is an estimate of the normality and our decision function.

Random partitioning produces notably shorter paths for anomalies. Hence when a forest produces shorter path lengths, these are very likely outliers in certain subsets of features.


### 2.7.2 Novelty detection method
- consider a distribution of n samples represented by p features. With incoming new data, does it originate from the already known distribution or is it from a previously unknown distribution?
- learn a rough, close boundary or decision function or frontier to enclose the distribution of points enclosed.
- if further observation lie within this boundary, it is considered as coming from this distribution

### 2.7.3 Outlier detection
- One has already a training set containing inlier and outliers.
- No clean dataset to start from.
- covariance.EllipticEnvelope used to estimate a robust covariante (Mahalanobis distance)
- assumes Gaussian distributed variables
- covariance is not robust against outliers

### 2.7.3.2 Isolation Forest
see above.
Maximul depth of tree: log_2(n), where n is the number of samples.

### 2.7.3.3 LocalOutlierFactor
- estimate local density using k-means algorithm
- number k of neighbors considered: k=20 seems to be a good number
- how isolated is it wrt the surrounding neighborhood
- no ```predict()```, ```score_samples_``` or ```decision_function```, just ```fit_predict()```

### Mahalanobis distance
- distance of one observation to the modes of its distribution in an n-dimensional space.
- how many standard deviation is an observation away from a  distribution D?

