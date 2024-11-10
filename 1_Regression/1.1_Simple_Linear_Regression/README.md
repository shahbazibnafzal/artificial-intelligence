# Simple Linear Regression

Simple linar regression is applied to predict a continuous value when it is dependant on a single variable.

**Equation:**

```math
y = b_{0} + b_{1}X
```

> y = dependant variable,
> x = independant variable,
> m = slope,
> c = intercept

The equation is for a simple line.

**How do we get the line of best fit?**

We calculate the ordinary least squared value and for that we calculate the residual for each datapoint in the lines and get the sum of all the residual. The line for which the sum of square of residuals is minnimum, is called the line of best fit.

```math
residual = y_{i} − \hat{y}{i}
```

```math
\hat{y} = b_{0} + b_{0}X_{1}

m, c sunch that:
SUM(y_{i} - \hat{y}{i}) is minimized
```