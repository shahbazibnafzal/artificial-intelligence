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
### Check null values

```python
dataset.isnull().sum()
```

### Splitting the dataset in training and test set

```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
```

### Training the simple linear regression model on training set

```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
```

### Predict the test set result

```python
Y_pred = regressor.predict(X_test)
```

### Visualizing the training set result

```python
plt.scatter(X_train, Y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs Experience (training set)")
plt.ylabel("Salary")
plt.xlabel("Expeience")
plt.show()
```

### Visualizing the test set result

```python
plt.scatter(X_test, Y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs Experience (test set)")
plt.ylabel("Salary")
plt.xlabel("Expeience")
plt.show()
```