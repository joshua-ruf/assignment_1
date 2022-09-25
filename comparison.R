
library(data.table)
library(magrittr)
library(ggplot2)

setwd('~/School/assignment_1/')

df0 = fread('problem_1/RESULTS.csv')
df0[, dataset := 'problem 1']

df1 = fread('problem_2/RESULTS.csv')
df1[, dataset := 'problem_2']

df = rbind(df0, df1)
df$full_train_accuracy = NULL
df$full_test_accuracy = NULL

(
  plot = df %>%
    .[!(model %in% c(
      "KNeighborsClassifier() [OVERSAMPLE]", 
      "DummyClassifier(random_state=0, strategy='most_frequent')"
    ))] %>%
    ggplot(aes(y = model)) +
    geom_point(aes(x = full_train_f1, color='training')) +
    geom_point(aes(x = full_test_f1, color='testing')) +
    theme(legend.position = 'bottom') +
    facet_wrap(~dataset) +
    scale_y_discrete(
      labels = function(x){
        gsub('\\(.*', '', x)
      }
    ) +
    theme_bw() +
    labs(
      title='Comparison of All Models across both datasets',
      x='F1 Score', y = NULL, color = NULL
    )
)

ggsave('final_figure.png', plot, width=8, height=3, dpi=300)
