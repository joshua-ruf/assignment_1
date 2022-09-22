
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

Overall the 6 month retention rate across these employees is XXX meaning that these data exhibit a rather high degree of class imbalance. As a consequence, many of the models described in more detail below, when run out-of-the-box merely guessed that all employees __would__ be employed until 6 months. As such, I chose to use the f1-score metric for model selection since it's more robust to class imbalance. The F1-metric jointly considers precision and recall, precision being the proportion of positive outputs that are truly positive, and recall being the proportion of positive examples that the model finds. This, in addition to including class weights as a parameter in `sklearn` and upsampling negative examples in `pytorch` managed to deal with the imbalance somewhat reasonably.

Ultimately I found that...

### Decision Tree


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