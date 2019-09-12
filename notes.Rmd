---
title: Cross-validation
date: 2019-01-28T13:30:00-06:00  # Schedule page publish date.
    
draft: false
type: docs

bibliography: [../../static/bib/sources.bib]
csl: [../../static/bib/apa.csl]
link-citations: true

menu:
  notes:
    parent: Resampling methods
    weight: 1
---

```{r setup, include = FALSE}
# set default chunk options
knitr::opts_chunk$set(cache = TRUE)
```

```{r packages, message = FALSE, warning = FALSE, cache = FALSE}
library(tidyverse)
library(tidymodels)
library(magrittr)
library(here)
library(rcfss)

set.seed(1234)
theme_set(theme_minimal())
```

\newcommand{\E}{\mathrm{E}} \newcommand{\Var}{\mathrm{Var}} \newcommand{\Cov}{\mathrm{Cov}} \newcommand{\se}{\text{se}} \newcommand{\Lagr}{\mathcal{L}} \newcommand{\lagr}{\mathcal{l}}

**Resampling methods** are essential to test and evaluate statistical models. Because you likely do not have the resources or capabilities to repeatedly sample from your population of interest, instead you can repeatedly draw from your original sample to obtain additional information about your model. For instance, you could repeatedly draw samples from your data, estimate a linear regression model on each sample, and then examine how the estimated model differs across each sample. This allows you to assess the variability and stability of your model in a way not possible if you can only fit the model once.

There are two major types of resampling methods we will consider:

* Cross-validation - frequently used for model assessment and evaluating a model's performance relative to other models
* [Bootstrap](/notes/bootstrap/) - commonly used to provide a non-parametric measure of the accuracy of a parameter estimate or a given statistical learning method

# Training/test set split

In most modeling situations, we can immediately partition the dataset into a **training** set and a **test** set. The training set will be used for model construction, and the test set will be used to evaluate the performance of the final model. This is most important -- while you can reuse the training set many times to build different statistical models, you can only use the test set of data once. If you reuse it, you introduce **data leakage** into your modeling process and no longer have unbiased estimates of the test error. This is why collaborative platforms such as [Kaggle](https://www.kaggle.com/) hold back a portion of the dataset in their competitions. You can use the training set to build the strongest performing model, but you cannot tune your model based on the test error because you do not have access to it.

# Validation set

Even accounting for the training/test set split, one issue with using the same data to both fit and evaluate our model is that we will bias our model towards fitting the data that we have. We may fit our function to create the results we expect or desire, rather than the "true" function. Instead, we can further split our training set into distinct **training** and **validation** sets. The training set can be used repeatedly to train different models. We then use the validation set to evaluate the model's performance, generating metrics such as the mean squared error (MSE) or the error rate. Unlike the test set, we are permitted to use the validation set multiple times. The important thing is that we do not use the validation set to train or fit the model, only evaluate its performance after it has been fit.

## Regression

Here we will examine the relationship between horsepower and car mileage in the `Auto` dataset (found in `library(ISLR)`):

```{r auto}
library(ISLR)

Auto <- as_tibble(Auto)
Auto
```

```{r auto_plot, dependson="auto"}
ggplot(Auto, aes(horsepower, mpg)) +
  geom_point()
```

The relationship does not appear to be strictly linear:

```{r auto_plot_lm, dependson="auto"}
ggplot(Auto, aes(horsepower, mpg)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)
```

Perhaps by adding quadratic terms to the linear regression we could improve overall model fit. To evaluate the model, we will split the data into a training set and validation set,^[For educational purposes, here we will omit the test set. In a real-world situation, we would first partition out a test set of data.] estimate a series of higher-order models, and calculate a test statistic summarizing the accuracy of the estimated `mpg`. To calculate the accuracy of the model, we will use [mean squared error](/notes/model-accuracy/#quality-of-fit) (MSE), defined as

$$MSE = \frac{1}{N} \sum_{i = 1}^{N}{(y_i - \hat{f}(x_i))^2}$$

For this task, first we use `rsample::initial_split()` to create training and validation sets (using a 50/50 split), then estimate a linear regression model without any quadratic terms.

* I use `set.seed()` in the beginning - whenever you are writing a script that involves randomization (here, random subsetting of the data), always set the seed at the beginning of the script. This ensures the results can be reproduced precisely.^[The actual value you use is irrelevant. Just be sure to set it in the script, otherwise R will randomly pick one each time you start a new session.]
* I also use the `glm()` function rather than `lm()` - if you don't change the `family` parameter, the results of `lm()` and `glm()` are exactly the same.^[The default `family` for `glm()` is `gaussian()`, or the **Gaussian** distribution. You probably know it by its other name, the [**Normal** distribution](https://en.wikipedia.org/wiki/Normal_distribution).]

```{r auto_split}
set.seed(1234)

auto_split <- initial_split(data = Auto, prop = 0.5)
auto_train <- training(auto_split)
auto_test <- testing(auto_split)
```

```{r auto_lm, dependson="auto_split"}
auto_lm <- glm(mpg ~ horsepower, data = auto_train)
summary(auto_lm)
```

To estimate the MSE for a single partition (i.e. for a training or validation set):

1. Use `broom::augment()` to generate predicted values for the data set
1. Calculate the MSE from the fitted values using `rcfss::mse()`.^[Note that most other metric functions can be found in the [`yardstick`](https://tidymodels.github.io/yardstick/) package. However `yardstick` only implements root mean squared error (RMSE), so I wrote a function in `rcfss` to calculate MSE. Otherwise you could just calculate RMSE and square the resulting value.]

For the training set, this would look like:

```{r mse-train, dependson = "auto_lm"}
(train_mse <- augment(auto_lm, newdata = auto_train) %>%
  mse(truth = mpg, estimate = .fitted))
```

For the validation set:

```{r mse-test, dependson = "auto_lm"}
(test_mse <- augment(auto_lm, newdata = auto_test) %>%
  mse(truth = mpg, estimate = .fitted))
```

For a strictly linear model, the MSE for the validation set is `r formatC(test_mse$.estimate[[1]], digits = 4)`. How does this compare to a quadratic model? We can use the `poly()` function in conjunction with a `map()` iteration to estimate the MSE for a series of models with higher-order polynomial terms:

```{r mse-poly, dependson = "auto_split"}
# visualize each model
ggplot(Auto, aes(horsepower, mpg)) +
  geom_point(alpha = .1) +
  geom_smooth(aes(color = "1"),
              method = "glm",
              formula = y ~ poly(x, i = 1, raw = TRUE),
              se = FALSE) +
  geom_smooth(aes(color = "2"),
              method = "glm",
              formula = y ~ poly(x, i = 2, raw = TRUE),
              se = FALSE) +
  geom_smooth(aes(color = "3"),
              method = "glm",
              formula = y ~ poly(x, i = 3, raw = TRUE),
              se = FALSE) +
  geom_smooth(aes(color = "4"),
              method = "glm",
              formula = y ~ poly(x, i = 4, raw = TRUE),
              se = FALSE) +
  geom_smooth(aes(color = "5"),
              method = "glm",
              formula = y ~ poly(x, i = 5, raw = TRUE),
              se = FALSE) +
  scale_color_brewer(type = "qual", palette = "Dark2") +
  labs(x = "Horsepower",
       y = "MPG",
       color = "Highest-order\npolynomial")

# function to estimate model using training set and generate fit statistics
# using the test set
poly_results <- function(train, test, i) {
  # Fit the model to the training set
  mod <- glm(mpg ~ poly(horsepower, i, raw = TRUE), data = train)
  
  # `augment` will save the predictions with the test data set
  res <- augment(mod, newdata = test) %>%
    mse(truth = mpg, estimate = .fitted)
  
  # Return the test data set with the additional columns
  res
}

# function to return MSE for a specific higher-order polynomial term
poly_mse <- function(i, train, test){
  poly_results(train, test, i) %$%
    mean(.estimate)
}

cv_mse <- tibble(terms = seq(from = 1, to = 5),
                 mse_test = map_dbl(terms, poly_mse, auto_train, auto_test))

ggplot(cv_mse, aes(terms, mse_test)) +
  geom_line() +
  labs(title = "Comparing quadratic linear models",
       subtitle = "Using validation set",
       x = "Highest-order polynomial",
       y = "Mean Squared Error")
```

Based on the MSE for the validation set, a polynomial model with a quadratic term ($\text{horsepower}^2$) produces a lower average error than the standard model. A higher order term such as a fifth-order polynomial leads to an even larger reduction, though increases the complexity of interpreting the model.

## Classification

Recall our efforts to [predict passenger survival during the sinking of the Titanic](/notes/logistic-regression/).

```{r titanic_data, message = FALSE}
library(titanic)
titanic <- as_tibble(titanic_train) %>%
  mutate(Survived = factor(Survived))

titanic %>%
  head() %>%
  knitr::kable()
```

```{r age_woman_cross}
survive_age_woman_x <- glm(Survived ~ Age * Sex, data = titanic,
                           family = binomial)
summary(survive_age_woman_x)
```

We can use the same validation set approach to evaluate the model's accuracy. For classification models, instead of using MSE we examine the **error rate**. That is, of all the predictions generated for the validation set, what percentage of predictions are incorrect? The goal is to minimize this value as much as possible (ideally, until we make no errors and our error rate is $0%$).

```{r logit}
# function to convert log-odds to probabilities
logit2prob <- function(x){
  exp(x) / (1 + exp(x))
}
```

```{r accuracy_age_gender_x_test_set, dependson="age_woman_cross", message = FALSE}
# split the data into training and validation sets
titanic_split <- initial_split(data = titanic, prop = 0.5)

# fit model to training data
train_model <- glm(Survived ~ Age * Sex, data = training(titanic_split),
                   family = binomial)
summary(train_model)

# calculate predictions using validation set
x_test_accuracy <- augment(train_model, newdata = testing(titanic_split)) %>% 
  as_tibble() %>%
  mutate(.prob = logit2prob(.fitted),
         .pred = factor(round(.prob)))

# calculate test accuracy rate
accuracy(x_test_accuracy, truth = Survived, estimate = .pred)
```

This interactive model generates an error rate of `r formatC((1 - mean(accuracy(x_test_accuracy, truth = Survived, estimate = .pred)$.estimate)) * 100, digits = 3)`%. We could compare this error rate to alternative classification models, either other logistic regression models (using different formulas) or a tree-based method.

## Drawbacks to validation sets

There are two main problems with validation sets:

1. Validation estimates of the test error rates can be highly variable depending on which observations are sampled into the training and validation sets. See what happens if we repeat the sampling, estimation, and validation procedure for the `Auto` data set:

    ```{r auto_variable_mse, dependson = "mse-poly"}
    mse_variable <- function(Auto){
      auto_split <- initial_split(Auto, prop = 0.5)
      auto_train <- training(auto_split)
      auto_test <- testing(auto_split)
      
      cv_mse <- tibble(terms = seq(from = 1, to = 5),
                           mse_test = map_dbl(terms, poly_mse, auto_train, auto_test))
      
      return(cv_mse)
    }
    
    rerun(10, mse_variable(Auto)) %>%
      bind_rows(.id = "id") %>%
      ggplot(aes(terms, mse_test, color = id)) +
      geom_line() +
      labs(title = "Variability of MSE estimates",
           subtitle = "Using the validation set approach",
           x = "Degree of Polynomial",
           y = "Mean Squared Error") +
      theme(legend.position = "none")
    ```
    
    Depending on the specific training/validation split, our MSE varies by up to 5.

1. If you don't have a large data set, you'll have to dramatically shrink the size of your training set. Most statistical learning methods perform better with more observations - if you don't have enough data in the training set, you might overestimate the error rate in the test set.

# Leave-one-out cross-validation

An alternative method is **leave-one-out cross validation** (LOOCV). Like with the validation set approach, you split the data into two parts. However the difference is that you only remove one observation for the validation set, and keep all remaining observations in the training set. The statistical learning method is fit on the $N-1$ training set. You then use the held-out observation to calculate the $MSE = (Y_1 - \hat{Y}_1)^2$ which should be an unbiased estimator of the test error. Because this MSE is highly dependent on which observation is held out, **we repeat this process for every single observation in the data set**. Mathematically, this looks like:

$$CV_{(N)} = \frac{1}{N} \sum_{i = 1}^{N}{MSE_i}$$

This method produces estimates of the error rate that are approximately unbiased and are non-varying for a given dataset, unlike the validation set approach where the MSE estimate is highly dependent on the sampling process for training/validation sets. However it can have high variance because the $N$ "training sets" are so similar to one another. LOOCV is also highly flexible and works with any kind of predictive modeling.

Of course the downside is that this method is computationally difficult. You have to estimate $N$ different models - if you have a large $N$ or each individual model takes a long time to compute, you may be stuck waiting a long time for the computer to finish its calculations.

## LOOCV in linear regression

We can use the `loo_cv()` function in the `rsample` library to compute the LOOCV of any linear or logistic regression model. It takes a single argument: the data frame being cross-validated. For the `Auto` dataset, this looks like:

```{r loocv-data, dependson="Auto"}
loocv_data <- loo_cv(Auto)
loocv_data
```

Each element of `loocv_data$splits` is an object of class `rsplit`. This is essentially an efficient container for storing both the **analysis** data (i.e. the training data set) and the **assessment** data (i.e. the validation data set). If we print the contents of a single `rsplit` object:

```{r rsplit, dependson = "loocv-data"}
first_resample <- loocv_data$splits[[1]]
first_resample
```

This tells us there are `r dim(first_resample)[["analysis"]]` observations in the analysis set, `r dim(first_resample)[["assessment"]]` observation in the assessment set, and the original data set contained `r dim(first_resample)[["n"]]` observations. To extract the analysis/assessment sets, use `analysis()` or `assessment()` respectively:

```{r rsplit-extract}
training(first_resample)
assessment(first_resample)
```

Given this new `loocv_data` data frame, we write a function that will, for each resample:

1. Obtain the analysis data set (i.e. the $N-1$ training set)
1. Fit a linear regression model
1. Predict the validation data (also known as the **assessment** data, the $1$ validation set) using the `broom` package
1. Determine the MSE for each sample

```{r loocv-function, dependson = "Auto"}
holdout_results <- function(splits) {
  # Fit the model to the N-1
  mod <- glm(mpg ~ horsepower, data = analysis(splits))
  
  # `augment` will save the predictions with the holdout data set
  res <- augment(mod, newdata = assessment(splits)) %>%
    # calculate the metric
    mse(truth = mpg, estimate = .fitted)
  
  # Return the metrics
  res
}
```

This function works for a single resample:

```{r loocv-function-test, dependson = "loocv-function"}
holdout_results(loocv_data$splits[[1]])
```

To compute the MSE for each heldout observation (i.e. estimate the test MSE for each of the $N$ observations), we use the `map()` function from the `purrr` package to estimate the model for each training set, then calculate the MSE for each observation in each validation set:

```{r loocv, dependson = c("Auto", "loocv-function")}
loocv_data_poly1 <- loocv_data %>%
  mutate(results = map(splits, holdout_results)) %>%
  unnest(results) %>%
  spread(.metric, .estimate)
loocv_data_poly1
```

Now we can compute the overall LOOCV MSE for the data set by calculating the mean of the `mse` column:

```{r loocv-test-mse, dependson = c("Auto", "loocv-function")}
loocv_data_poly1 %>%
  summarize(mse = mean(mse))
```

We can also use this method to compare the optimal number of polynomial terms as before.

```{r loocv_poly, dependson="Auto"}
# modified function to estimate model with varying highest order polynomial
holdout_results <- function(splits, i) {
  # Fit the model to the N-1
  mod <- glm(mpg ~ poly(horsepower, i, raw = TRUE), data = analysis(splits))
  
  # `augment` will save the predictions with the holdout data set
  res <- augment(mod, newdata = assessment(splits)) %>%
    # calculate the metric
    mse(truth = mpg, estimate = .fitted)
  
  # Return the assessment data set with the additional columns
  res
}

# function to return MSE for a specific higher-order polynomial term
poly_mse <- function(i, loocv_data){
  loocv_mod <- loocv_data %>%
    mutate(results = map(splits, holdout_results, i)) %>%
    unnest(results) %>%
    spread(.metric, .estimate)
  
  mean(loocv_mod$mse)
}

cv_mse <- tibble(terms = seq(from = 1, to = 5),
                 mse_loocv = map_dbl(terms, poly_mse, loocv_data))
cv_mse

ggplot(cv_mse, aes(terms, mse_loocv)) +
  geom_line() +
  labs(title = "Comparing quadratic linear models",
       subtitle = "Using LOOCV",
       x = "Highest-order polynomial",
       y = "Mean Squared Error")
```

And arrive at a similar conclusion. There may be a very marginal advantage to adding a fifth-order polynomial, but not substantial enough for the additional complexity over a mere second-order polynomial.

## LOOCV in classification

Let's verify the error rate of our interactive terms model for the Titanic data set:

```{r titanic-loocv}
# function to generate assessment statistics for titanic model
holdout_results <- function(splits) {
  # Fit the model to the N-1
  mod <- glm(Survived ~ Age * Sex, data = analysis(splits),
             family = binomial)
  
  # `augment` will save the predictions with the holdout data set
  res <- augment(mod, newdata = assessment(splits)) %>% 
    as_tibble() %>%
    mutate(.prob = logit2prob(.fitted),
           .pred = round(.prob))
  
  # Return the assessment data set with the additional columns
  res
}

titanic_loocv <- loo_cv(titanic) %>%
  mutate(results = map(splits, holdout_results)) %>%
  unnest(results) %>%
  mutate(.pred = factor(.pred)) %>%
  group_by(id) %>%
  accuracy(truth = Survived, estimate = .pred)

1 - mean(titanic_loocv$.estimate, na.rm = TRUE)
```

In a classification problem, the LOOCV tells us the average error rate based on our predictions. So here, it tells us that the interactive `Age * Sex` model has a `r formatC((1 - mean(titanic_loocv$.estimate, na.rm = TRUE)) * 100, digits = 3)`% error rate. This is similar to the validation set result (`r formatC((1 - mean(accuracy(x_test_accuracy, truth = Survived, estimate = .pred)$.estimate)) * 100, digits = 3)`%).

# $K$-fold cross-validation

A less computationally-intensive approach to cross validation is **$K$-fold cross-validation**. Rather than dividing the data into $N$ groups, one divides the observations into $K$ groups, or **folds**, of approximately equal size. The first fold is treated as the validation set, and the model is estimated on the remaining $K-1$ folds. This process is repeated $K$ times, with each fold serving as the validation set precisely once. The $K$-fold CV estimate is calculated by averaging the MSE values for each fold:

$$CV_{(K)} = \frac{1}{K} \sum_{i = 1}^{K}{MSE_i}$$

As you may have noticed, LOOCV is a special case of $K$-fold cross-validation where $K = N$. More typically researchers will use $K=5$ or $K=10$ depending on the size of the data set and the complexity of the statistical model.

## $K$-fold CV in linear regression

Let's go back to the `Auto` data set. Instead of LOOCV, let's use 10-fold CV to compare the different polynomial models.

```{r 10_fold_auto}
# modified function to estimate model with varying highest order polynomial
holdout_results <- function(splits, i) {
  # Fit the model to the training set
  mod <- glm(mpg ~ poly(horsepower, i, raw = TRUE), data = analysis(splits))
  
  # Save the heldout observations
  holdout <- assessment(splits)
  
  # `augment` will save the predictions with the holdout data set
  res <- augment(mod, newdata = holdout) %>%
    # calculate the metric
    mse(truth = mpg, estimate = .fitted)
  
  # Return the assessment data set with the additional columns
  res
}

# function to return MSE for a specific higher-order polynomial term
poly_mse <- function(i, vfold_data){
  vfold_mod <- vfold_data %>%
    mutate(results = map(splits, holdout_results, i)) %>%
    unnest(results) %>%
    spread(.metric, .estimate)
  
  mean(vfold_mod$mse)
}

# split Auto into 10 folds
auto_cv10 <- vfold_cv(data = Auto, v = 10)

cv_mse <- tibble(terms = seq(from = 1, to = 5),
                     mse_vfold = map_dbl(terms, poly_mse, auto_cv10))
cv_mse
```

How do these results compare to the LOOCV values?

```{r 10_fold_auto_loocv, dependson=c("10_fold_auto","loocv_poly")}
auto_loocv <- loo_cv(Auto)

tibble(terms = seq(from = 1, to = 5),
       `10-fold` = map_dbl(terms, poly_mse, auto_cv10),
       LOOCV = map_dbl(terms, poly_mse, auto_loocv)
) %>%
  gather(method, MSE, -terms) %>%
  ggplot(aes(terms, MSE, color = method)) +
  geom_line() +
  scale_color_brewer(type = "qual") +
  labs(title = "MSE estimates",
       x = "Degree of Polynomial",
       y = "Mean Squared Error",
       color = "CV Method")
```

Pretty much the same results.

## Computational speed of LOOCV vs. $K$-fold CV

### LOOCV

```{r loocv_time}
library(tictoc)

tic()
poly_mse(vfold_data = auto_loocv, i = 2)
toc()
```

### 10-fold CV

```{r kfold_time}
tic()
poly_mse(vfold_data = auto_cv10, i = 2)
toc()
```

On my machine, 10-fold CV was about 40 times faster than LOOCV. Again, estimating $K=10$ models is going to be much easier than estimating $K=`r nrow(Auto)`$ models.

## $K$-fold CV in logistic regression

You've gotten the idea by now, but let's do it one more time on our interactive Titanic model.

```{r titanic_kfold}
# function to generate assessment statistics for titanic model
holdout_results <- function(splits) {
  # Fit the model to the training set
  mod <- glm(Survived ~ Age * Sex, data = analysis(splits),
             family = binomial)
  
  # `augment` will save the predictions with the holdout data set
  res <- augment(mod, newdata = assessment(splits)) %>% 
    as_tibble() %>%
    mutate(.prob = logit2prob(.fitted),
           .pred = round(.prob))

  # Return the assessment data set with the additional columns
  res
}

titanic_cv10 <- vfold_cv(data = titanic, v = 10) %>%
  mutate(results = map(splits, holdout_results)) %>%
  unnest(results) %>%
  mutate(.pred = factor(.pred)) %>%
  group_by(id) %>%
  accuracy(truth = Survived, estimate = .pred)

1 - mean(titanic_cv10$.estimate, na.rm = TRUE)
```

Not a large difference from the LOOCV approach, but it take much less time to compute.

# Appropriate value for $K$

Ignoring the computational efficiency concerns, why not always estimate cross-validation with $K=N$? Or more generally, what is the optimal value for $K$? **It depends.** Well that is not very helpful.

With more explanation, it depends on how we wish to handle the bias-variance tradeoff. LOOCV is a low-bias, high-variance method. That is, it provides unbiased estimates of the test error since each training set contains $N-1$ observations. This is almost as many observations as contained in the full data set. $K$-fold CV for $K=5$ or $10$ leads to an intermediate amount of bias, since each training set contains $\frac{(K-1)N}{K}$ observations. This is fewer than LOOCV, but more than a standard validation set approach with just a single split into training and validation sets.

The amount of bias is also driven by the size of the training set. The larger the training set, the less bias we should expect in our results because the model was fit to more data points. Consider a hypothetical learning curve for a classifier:

```{r hypo-class, echo = FALSE}
tibble(x = c(0, 200),
       y = c(0, 1)) %>%
  ggplot(aes(x, y)) +
  stat_function(fun = function(x) y = - (-0.03342708/0.04196558) * (1 - exp(-0.04196558 * x))) +
  labs(title = "Hypothetical learning curve for a classifier",
       x = "Size of training set",
       y = "1 - Expected test error")
```

As the training set size increases, $1 - \text{Expected test error}$ increases as well with diminishing returns. With 200 observations, $5$-fold cross-validation would use training sets of 160 observations, which is fairly similar to the full set. However with a dataset of 50, $5$-fold cross-validation would use training sets of 40 observations which has more error. Therefore the larger the data set, the fewer folds you can implement.

If all we care about is bias, we should prefer LOOCV. However, recall the contributors to a model's error:

$$\text{Error} = \text{Irreducible Error} + \text{Bias}^2 + \text{Variance}$$

We also should be concerned with the **variance** of the model. LOOCV has a higher variance than $K$-fold with $K < N$. When we perform LOOCV, we are averaging the outputs of $N$ fitted models which are trained on nearly entirely identical sets of observations. The results will be highly correlated with one another. In contrast, $K$-fold CV with $K < N$ averages the output of $K$ fitted models that are less correlated with one another, since the data sets are not as identical. Since the mean of many highly correlated quantities has higher variance than the mean of quantities with less correlation, the test error estimate from LOOCV has higher variance than the test error estimate from $K$-fold CV.

Given these considerations, a typical approach uses $K=5$ or $K=10$. Empirical research (see @breiman1992submodel, @kohavi1995study) shows that cross-validation with these number of folds suffers neither excessively high bias nor excessively high variance.

# Variations on cross-validation

To ensure each set is approximately similar to one another in every important aspect, we use random sampling without replacement to partition the data set. Alternative approaches include:

* Stratified cross-validation - splitting the data into folds based on criteria such as ensuring each fold has the same proportion of observations with a given categorical value (such as the response class value). While random sampling should lead to approximately equal class balances, stratified sampling will ensure an equal balance across folds.
* Repeated cross-validation - this is where $K$-fold CV is repeated $N$ times, where for each $N$ the data sample is shuffled prior to each repetition. This ensures a different split of the sample.
* Cross-validation with time series data - @bergmeir2012use evaluate multiple forms of cross-validation methods for time series models. For some types of models, standard cross-validation techniques can be employed without bias. In other situations, a standard approach is to partition the data temporally. For instance, if you have 10 years of observations for a given unit you may reserve the first 8 years for the training set and the last 2 years for the validation set. Other forms of cross-validation use **rolling validation sets** whereby one generates a series of validation sets, each containing a single observation. The training set consists only of observations that occurred prior to the observation that forms the validation set.

    ![From [Cross-validation for time series](https://robjhyndman.com/hyndsight/tscv/)](https://robjhyndman.com/files/cv1-1.png)

# Session Info {.toc-ignore}

```{r child = here::here("R", "_session-info.Rmd")}
```

# References {.toc-ignore}

* @james2013introduction
* @friedman2001elements
