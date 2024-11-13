# Multiple Linear Regression

Multiple Linear regression is used to predict continuous value when there are multiple independant variables.

**Equation**

```math
\hat{y} =  b_{0} + b_{1}X_{1} + b_{2}X_{2} + b_{3}X_{3} + ... + b_{n}X_{n}
```

# Assumption of linear regression

There are 5 linear regressions that need to be fulfilled in order to apply multiple linear regression to a dataset.

1. Linearity: Linearity between X and Y
2. Homoscedasticity: Equal variance
3. Multivariate Normality: Normality of error distribution
4. Independence: Observations should not have auto correlation (rows)
5. Lack of Multicollinearity: Predictors are not correlated with each others.

## Dummy variables

Whenever you have a categorical variable, turn them into dummy variable.
For example, if you have a column called state and the value is UP and Delhi then you can create two dummy variables called UP and Delhi and put the value as 0 as 1 and discard the state column.

## Dummy variable trap

Always include 1 less dummy variable than the actual number of dummy variables.