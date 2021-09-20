# Machine Learning Algorithms

##### import pandas as pd

##### import numpy as np

## Supervised

- Decision Trees
- Random Forests
- Gradient Boosts
- SVM

* ### Classification

  - LDA

* ### Regression

## Unsupervised

- ### Dimension Reduction
  - PCA
- ### Clustering
  - Partitional Algorithms (K-Means)
  - Hierarchical Clustering
- ### Association

---

## Linear Regression

Linear regression is the most basic type of regression commonly used for
predictive analysis. The idea is pretty simple: we have a dataset and we have
features associated with it. Features should be chosen very cautiously
as they determine how much our model will be able to make future predictions.
We try to set the weight of these features, over many iterations, so that they best
fit our dataset. We try to best fit a line through dataset and estimate the parameters.

##### from sklearn.model_selection import train_test_split

- import packages
- load and inspect the data
- plot data (scatter)
- train test split
- implement a least squares function to find a, b
- plot the line with the train data
- Classify the test data in to classes
- plot the line with each class so we can clearly see the split
- Get the total error

---

## Logistic Regression

Logistic regression is a generalized linear model that we can use to model or predict categorical outcome variables. We might use logistic regression to predict whether someone will be denied or approved for a loan, but probably not to predict the value of someone's house.
In logistic regression, we're essentially trying to find the weights that maximize the likelihood of producing our given data.

##### from sklearn.linear_model import LogisticRegression

- import packages
- load data
- plot
- sigmoid function
- Calculating the Log-Likelihood (sum over all the training data)
- Building the Logistic Regression Function
- time to do the regression
- print weights
- Comparing to Sk-Learn's LogisticRegression
- implement sklearn logistic regresion and fit
- print clf
- accuray
- plot results

---

## PCA

Principal Component Analysis (**PCA**) is statistical procedure that uses an
orthogonal transformation to convert set observations of possibly correlated
variables into a set of values of linear uncorrelated variables called
principal components.

##### from sklearn import model_selection

##### from sklearn.preprocessing import StandardScaler

##### from sklearn.decomposition import PCA

##### from sklearn import preprocessing

##### from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score

- Import needed packages
- Set random seed
- Load the Iris dataset included with scikit-learn
- Put data in a pandas DataFrame
- Add target and class to DataFrame
- start PCA
- Plot a graph to visualize
- Run the PCA model
- fit transform dataframe
- compare it with the original dataframe and to what it corresponds
- plot it

---

## LDA

**LDA** is a classification method using linear combination of variables

##### from sklearn import model_selection

##### from sklearn.preprocessing import StandardScaler

##### from sklearn import preprocessing

##### from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score

##### from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

- import packages
- load dataset
- Put data in a pandas DataFrame
- Add target and class to DataFrame
- implement the LDA
- identify your X's and your y's
- train-test-split
- Scale the X's
- apply the lda transformation
- Run a Random Forest Classifier with the transformed data
- Check the new acuracy! Is it higher or lower than what you obtain selecting 2 features and applying a Random Forest Directly?

### LDA X PCA

**LDA** is a classification method using linear combination of variables while **PCA** is a dimension reduction method.

- **PCA** (Not considering labels, unsupervised)
- **LDA** (Considering labels, supervised)

### Covariance != Correlation

**COVARIANCE**

- Positive - both variables grow
- Negative - One grows, other decreases
- Zero - Variables are independent

**Covariance Matrix** is the covariance of each of the variables of a dataset against all the others.

- **Covariance**: a measure used to indicate the extent to which two random variables change in tandem. Covariance is a way to measure correlation.
- **Correlation**: a measure used to represent how strongly two random variables are related.
- **Covariance** goes from -∞ to +∞
- **Correlation** goes from -1 to 1
- **Covariance** is affecte by a change in scale. **Correlation** is not.

---

## K-Means

##### from sklearn.cluster import KMeans

##### from sklearn.model_selection import train_test_split

##### from scipy.cluster.hierarchy import linkage, dendrogram

- import packages
- load data
- create pandas dataframe
- Create the class and target columns
- clean data
- merge
- plot
- start k-means
- model
- fit
- predict
- plot (points)
- new labels
- predict new labels
- plot new labels (new_points)
- plot all together

---

## KNN

The k-nearest neighbors algorithm (knn) is a non-parametric method used for classification and regression. The KNN algorithm treats the features as coordinates in a multidimensional feature space.

#### Advantages of KNN

- Intuitive and simple
- Has no assumptions
- no training step
- variety of distance criteria to be choose from
- constantly evolves
- "easy" to implement for multi-class problem

#### Disadvantages of KNN

- slow algorithm
- curse of dimensionality
- optimal number of neighbors
- outlier sensitivity
- imbalanced data causes problems
- missing value treatment

KNN

- import libraries
- make the dataset
- plot the dataset
- implement square diff
- implement root sum squares
- euclidean_distances
- evaluate
- Create the KNN prediction function
- fit
- model
- train test split
- predict

---

## Decision Trees

Decision tree is one of the predictive modelling approaches used in statistics, data mining and machine learning.

- each internal node represents a test on a feature (e.g. whether a coin flip comes up heads or tails)
- each leaf node represents a class label
- branches represent conjunctions of features that lead to those class labels

A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements. Wikipedia

- import packages
- load data
- create pandas dataframe
- train test split
- DecisionTreeClassifier()
- fit
- predict
- accuracy

---

## Random Forests

Random Forests conbine versatility and power into a single machine-learning approach.

Random Forest is a model made up of many decision trees.

**Random Forests** are the result of **combining multiple trees** that are **not correlated** with each other.

##### from sklearn.ensemble import RandomForestClassifier

- load data
- create pandas dataframe
- train test split
- RandomForestClassifier()
- fit
- predict
- accuracy

---

## Gradient Boost - XG Boost

##### from xgboost import XGBRegressor

- import packages
- load data
- create pandas dataframe
- Remove rows with missing target, separate target from predictors
- train test split (Break off validation set from training data)
- Cardinality
- Select numeric columns
- one hot encode
- Define the model - XGBRegressor()
- fit
- predict
- Calculate MAE

---

## SVM - Support Vector Machines

##### from sklearn.svm import SVC

##### from sklearn.metrics import accuracy_score

##### from sklearn.model_selection import GridSearchCV

Support vector Machine (SVM) is a supervised machine learning algorithm that can be used for classification and regression. The objective is to find the best splitting boundary between data. The goal is to create a **flat boundary** called **hyperplane**, which divides the space to create fairly **homogeneous partitions** on either side.

Support vectors are the points from each class that are the closest to the **Maximum Margin Hyperplane (MMH)**. Each class must have at least one support vector, but ot is possible to have more than one.

- import packages
- load data
- create pandas dataframe
- train test split
- Train the support vector classifier - SVC()
- predict
- accuracy_score()
- get the confusion matrix

---
