
library(data.table)
library(magrittr)
library(ggplot2)

setwd('~/School/assignment_1/problem_2/')

# tree
df = fread('predicted-gm-non-ebert.csv')

df$gm_predicted %>% mean()
# 20.9% labelled as great movies

df[order(-gm_predicted_prob)] %>%
  head(50)

D = df[, .(pct_gm = mean(gm_predicted)), reviewer] %>%
  .[order(pct_gm)]

D[, reviewer := factor(reviewer, as.character(D$reviewer))]

(
  plot = D[pct_gm > 0] %>%
    ggplot(aes(x = pct_gm, y = reviewer)) +
    geom_col(fill='dodgerblue4', color='black') +
    theme_minimal() +
    scale_x_continuous(labels=function(x){sprintf("%s%%", x*100)}) +
    labs(title='Reviewers with the highest percentage of predicted "great movies"')
)

ggsave('pct_gm_by_reviewer.png', plot, width=7, height=4, dpi=200)


D = df[, .(pct_gm = mean(gm_predicted)), stars] %>%
  .[order(pct_gm)]

(
  plot = D[pct_gm > 0] %>%
    ggplot(aes(x = pct_gm, y = as.character(stars))) +
    geom_col(fill='dodgerblue4', color='black') +
    theme_minimal() +
    scale_x_continuous(labels=function(x){sprintf("%s%%", x*100)}) +
    labs(y='stars',
         title='Star Ratings with the highest percentage of predicted "great movies"'
         )
)

ggsave('pct_gm_by_stars.png', plot, width=7, height=4, dpi=200)


D = df[, .(title, link, gm_predicted_prob)]

D[order(-gm_predicted_prob, title), sprintf("[%s](%s), p=%.4f", title, link, gm_predicted_prob)] %>%
  cat(sep = "\n * ", file='predicted_great_movies.md')

(
  plot = D[pct_gm > 0] %>%
    ggplot(aes(x = pct_gm, y = as.character(stars))) +
    geom_col(fill='dodgerblue4', color='black') +
    theme_minimal() +
    scale_x_continuous(labels=function(x){sprintf("%s%%", x*100)}) +
    labs(y='stars')
)





