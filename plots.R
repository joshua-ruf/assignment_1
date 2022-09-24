
library(data.table)
library(magrittr)
library(ggplot2)

setwd('~/School/assignment_1/')

# adaboost
df = fread('AdaBoostClassifier.csv')

plot = df[
  param_base_estimator__splitter=='random' & param_base_estimator__max_features == 'sqrt'
] %>%
  ggplot(aes(x = param_n_estimators, y = mean_test_score)) +
  geom_point() +
  geom_smooth(method='lm', formula = y~poly(x, 2)) +
  labs(
    title='Mean F1-core amongst cross validation folds by number of estimators',
    caption='limited to splitter=="random" and max_features=="sqrt"'
  )

ggsave('adaboost_n_estimators.png', plot, width=8, height=3, dpi=200)


### SVC

df=fread('SVC.csv')

print(df[rank_test_score==1][1]$params)

(plot = df[param_kernel != 'linear'] %>%
    # .[param_kernel == 'poly'] %>%
  ggplot(aes(x = param_coef0,
             y = mean_test_score,
             color = param_kernel)) +
  geom_point() + 
  geom_smooth(
    alpha=0.2,
    method = 'lm', formula = y~poly(x, 2)) +
  labs(title='Mean score vs. coef, by kernel function') +
  theme_minimal())


ggsave('svc_score_by_coef.png', plot, width=8, height=3, dpi=200)

(plot = df[param_kernel != 'linear'] %>%
    # .[param_kernel == 'poly'] %>%
    ggplot(aes(x = param_C,
               y = mean_test_score,
               color = param_kernel)) +
    geom_point() + 
    geom_smooth(
      alpha=0.2,
      method = 'lm', formula = y~poly(x, 2)) +
    labs(title='Mean score vs. coef, by kernel function') +
    theme_minimal())


ggsave('svc_score_by_C.png', plot, width=8, height=3, dpi=200)


############### KNN
df=fread('KNN.csv')

print(df[rank_test_score==1][1]$params)

(plot = df %>%
    # .[param_kernel == 'poly'] %>%
    ggplot(aes(x = param_n_neighbors,
               y = mean_test_score,
               color = param_weights)) +
    geom_point() + 
    geom_smooth(
      aes(y = mean_train_score),
      linetype='dotted'
    ) +
    geom_smooth(
      alpha=0.2,
      method = 'lm', formula = y~poly(x, 2)) +
    labs(title='KNN: Mean score vs. k') +
    theme_minimal() +
    guides(color='none'))

ggsave('knn_score_by_k.png', plot, width=8, height=3, dpi=200)

df=fread('KNN_oversample.csv')

print(df[rank_test_score==1][1]$params)

(plot = df %>%
    # .[param_kernel == 'poly'] %>%
    ggplot(aes(x = param_n_neighbors)) +
    geom_point(aes(y = mean_test_score,
                   color = 'mean test score')) +
    geom_smooth(aes(y = mean_test_score,
                   color = 'mean test score')) +
    geom_smooth(
      aes(y = mean_train_score,
          color = 'mean train score'),
    ) +
    geom_point(
      aes(y = mean_train_score,
          color = 'mean train score'),
    ) +
    labs(title='KNN (oversampled): Mean score vs. k') +
    theme_minimal() +
    facet_wrap(~param_weights))

ggsave('knn_oversample_score_by_k.png', plot, width=8, height=3, dpi=200)

