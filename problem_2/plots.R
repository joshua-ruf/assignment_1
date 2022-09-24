
library(data.table)
library(magrittr)
library(ggplot2)

setwd('~/School/assignment_1/problem_2/')

# tree
df = fread('DecisionTreeClassifier.csv')
print(df[rank_test_score==1][1]$params)

(
  plot = df %>%
    ggplot(aes(x = param_max_depth, y = mean_test_score)) +
    geom_point() +
    geom_smooth(method='lm', formula = y~poly(x, 2)) +
    labs(
      title='DecisionTree: Mean F1-core by max depth and splitter',
    ) + 
    theme_minimal() +
    facet_wrap(~param_splitter)
)

# best splitter much better because lack of noise and high dimension
ggsave('dtc_max_depth_and_splitter.png', plot, width=8, height=3, dpi=200)


# adaboost
df = fread('AdaBoostClassifier.csv')

print(df[rank_test_score==1][1]$params)

(
  plot = df %>%
  ggplot(aes(x = param_n_estimators, y = mean_test_score)) +
  geom_point() +
  geom_smooth(method='lm', formula = y~poly(x, 2)) +
  labs(
    title='Boosting: Mean F1-core amongst cross validation folds by number of estimators',
  ) + 
  theme_minimal()
)

ggsave('adaboost_n_estimators.png', plot, width=8, height=3, dpi=200)


### SVC

df=fread('SVC.csv')

print(df[rank_test_score==1][1]$params)

(
  plot = df %>%
  ggplot(aes(x = param_coef0,
             y = mean_test_score,
             color = param_kernel)) +
  geom_point() + 
  geom_smooth(
    alpha=0.2,
    method = 'lm', formula = y~poly(x, 2)) +
  labs(title='SVC: Mean score vs. coef, by kernel function') +
  theme_minimal()
)

ggsave('svc_score_by_coef.png', plot, width=8, height=3, dpi=200)

(
  plot = df %>%
    ggplot(aes(x = param_C,
               y = mean_test_score,
               color = param_kernel)) +
    geom_point() + 
    geom_smooth(
      alpha=0.2,
      method = 'lm', formula = y~poly(x, 2)) +
    labs(title='SVC: Mean score vs. regularization parameter, by kernel function') +
    theme_minimal()
)

#### higher C is better because less noise
ggsave('svc_score_by_C.png', plot, width=8, height=3, dpi=200)


############### KNN
df=fread('KNN.csv')

print(df[rank_test_score==1][1]$params)

# (plot = df %>%
#     ggplot(aes(x = param_n_neighbors)) +
#     geom_point() + 
#     geom_smooth(
#       aes(y = mean_train_score, color = 'mean_test_score'),
#     ) +
#     geom_smooth(
#       alpha=0.2,
#       method = 'lm', formula = y~poly(x, 2)) +
#     labs(title='KNN: Mean score vs. k') +
#     theme_minimal() +
#     guides(color='none'))
# 
# ggsave('knn_score_by_k.png', plot, width=8, height=3, dpi=200)

df=fread('KNN_oversample.csv')

print(df[rank_test_score==1][1]$params)

(plot = df %>%
    .[param_weights == 'uniform'] %>%
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
    labs(title='KNN: Train vs. Test by k and distnace metric') +
    theme_minimal()
  )

# many more neighbours
ggsave('knn_test_vs_train_by_k.png', plot, width=8, height=3, dpi=200)


####### NNET

df = fread('NNet.csv')

df[order(best_vloss)][1]

(
  plot = df %>%
    ggplot(aes(x = epochs)) +
    geom_point(aes(y=training_f1_score, color='training')) +
    geom_point(aes(y=testing_f1_score, color='testing')) +
    geom_smooth(aes(y=training_f1_score, color='training'), method='lm') +
    geom_smooth(aes(y=testing_f1_score, color='testing'), method='lm') +
    facet_wrap(~paste("Hidden Dim:", hidden_dim)) +
    theme_minimal() +
    labs(
      color=NULL,
      title="NNet: Train vs Test F1-score by number of epochs and hidden dimension",
      y='F1 Score'
    )
)

ggsave('nnet_train_vs_test_by_hidden_dims.png', plot, width=8, height=4, dpi=200)


(
  plot = df %>%
    ggplot(aes(x = epochs)) +
    geom_point(aes(y=training_f1_score, color='training')) +
    geom_point(aes(y=testing_f1_score, color='testing')) +
    geom_smooth(aes(y=training_f1_score, color='training'), method='lm') +
    geom_smooth(aes(y=testing_f1_score, color='testing'), method='lm') +
    facet_wrap(~paste("Batch Size:", batch_size)) +
    theme_minimal() +
    labs(
      color=NULL,
      title="Train vs Test F1-score by number of epochs and batch size",
      y='F1 Score'
    )
)

ggsave('nnet_train_vs_test_by_batch_size.png', plot, width=8, height=4, dpi=200)

# larger batch size helps
# 8 hidden dim seems best, maybe 16 => 4 and no real improvement but 32 is way overfit

(
  plot = df %>%
    ggplot(aes(x = epochs)) +
    geom_point(aes(y=training_f1_score, color='training')) +
    geom_point(aes(y=testing_f1_score, color='testing')) +
    geom_smooth(aes(y=training_f1_score, color='training'), method='lm') +
    geom_smooth(aes(y=testing_f1_score, color='testing'), method='lm') +
    facet_wrap(~paste("Dropout Rate:", dropout_rate)) +
    theme_minimal() +
    labs(
      color=NULL,
      title="Train vs Test F1-score by number of epochs and dropout_rate",
      y='F1 Score'
    )
)


(
  plot = df %>%
    ggplot(aes(x=hidden_dim)) +
    geom_point(aes(y=dropout_rate, fill=testing_f1_score))
    # geom_point(aes(y=testing_f1_score, color='testing')) +
    # geom_smooth(aes(y=training_f1_score, color='training'), method='lm') +
    # geom_smooth(aes(y=testing_f1_score, color='testing'), method='lm')
)
