# Simple Linear Regression

Simple linar regression is applied to predict a continuous value when it is dependant on a single variable and the relation between the two variables is linear.

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
```

m, c such that:

```math
SUM(y_{i} - \hat{y}_{i})
```

is minimized.

## Applying simple linear regression to Salary_Data.csv dataset to predict the salary of an employee.

### Importing the libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

### Importing the dataset

```python
dataset = read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
```
