
#### CS-7641 Assignment 1
#### Joshua Ruf

## Classification Problem 1

### Intro

The first classification problem is to predict whether an employee will stay in their new job for at least 6 months. I work for an HR Analytics company that builds pre-hire survey modules in an effort to help our clients hire better. Employee retention is very important to our clients and we're always looking for questions or sets of questions that give an indication as to whether a new hire will be engaged on the job. Practically, clients are interested in predicting whether a new hire will churn so they can engage in better workforce planning.

These data come from a single client, anonymized to keep the organization and the individual employees hidden. The data covers all employees hired between 2018-01-01 and 2022-03-01 so that a full 6 months of data will be available for each employee. Including all hires up until today would be a censored data problem whereby some employees are still employed but have not yet worked 6 months, adding complexity beyond the scope of this assignment.

The data includes a few sets of variables:

1. Basic demographic features such as age, ethnicity, and gender (where ethnicity and gender were coded as a series of binary variables)
2. Whether they are a manager and whether they were a referral (both binary)
3. Self-assessment overall score, and reference-assessment overall score
4. Self-reported strengths (series of binary variables)
5. Self-reported weaknesses (series of binary variables)
6. Self-reported skills (rated 1-5)
7. And of course, whether the employee terminated in the first 6 months of employment

Overall the 6 month termination rate across these employees is 12.7% meaning that these data exhibits a rather high degree of class imbalance. As a consequence, many models out-of-the-box simply guessed that all employees __would__ be employed until 6 months. As such, I chose to use the f1-score metric for model selection since it's more robust to class imbalance. The F1-metric jointly considers precision and recall, precision being the proportion of positive outputs that are truly positive, and recall being the proportion of all positive examples that the model labels as positive. The `DummyClassifier` makes the tradeoffs between accuracy and f1-score clear: when always guessing the most frequent value the test accuracy is 0.874 but the f1-score is 0.0 (the lowest possible value) since no negative samples are found and recall is zero. Conversely, when choosing randomly between True and False (weighted by their relative prevalence in the data), the DummyClassifier achieves a lower test accuracy of roughly 0.783 but a higher test f1-score of 0.168. 

These scores will serve as useful benchmarks in comparing subsequent models. The bottom line is, if we are willing to sacrifice some accuracy then we may be able to increase the probability that our models can identify negative examples in our data.

Ultimately I found that...

### Decision Tree

The decision tree classifier, after 1500 rounds of 5-fold cross validation achieved a training f1-score of 0.377358 and a testing f1-score of 0.290909, well outperforming the stratified DummyClassifier. In cross-validation I played with the following hyperparameters:

1. `Criterion`: how to select the feature on which to split each node (entropy and gini). In either case the "impurity" at a given node is compared to the average impurity of its leaves, weighted by leaf size--the difference is called "information gain". The CART algorithm checks the every feature to determine which gives rise to the largest information gain and chooses that feature for splitting. In practice entropy and gini are very similar and this hyperparameter can largely be used as a sanity check to ensure other hyperparameters have a positive impact on model performance across numerous conditions.
2. `splitter`: whether to choose the feature with the highest information gain, or instead, to choose the feature at random, weighted by the relative information gain. This can be helpful to avoid overfitting as the relative ordering of information gain could be a quirk in the training data.
3. `ccp_alpha`: a kind of pruning, whereby there is a cost to adding nodes to the tree and after some threshold growing will cease.
4. `max_features`: whether to consider all features at each node, or instead a random sample of sqrt(number of features). Decreasing access to information will hurt training but can avoid overfitting.
5. `min_samples_split`: the fewest number of samples necessary to split a node. XXX
6. `min_samples_leaf`: the fewest number of samples necessary to be a leaf. Allowing larger leafs ensures that trees do not grow prohibitively large and that later splitting decisions are not so arbitrary.
7. `max_depth`: how many splits CART can perform before terminating.

In my data, criterion and splitter made little difference to performance, and limiting the number of features to consider in each split noticeably dropped performance. In theory, cpp_alpha, min_samples_split, min_samples_leaf, and max_depth all have similar goals of reducing tree size in an effort to minimize overfitting as larger trees tend to learn idiosyncrasies in the training data that do not generalize well to unseen data. In cross validation I saw them work together to rein in overfitting--the chosen model uses a combination of all four and visual analysis suggests that they are somewhat interchangeable.



### Boosting


### Support Vector Machines


### k-Nearest Neighbors


### Neural Networks



## Classification Problem 2


### Decision Tree


### Boosting


### Support Vector Machines


### k-Nearest Neighbors


### Neural Networks


## Resources

