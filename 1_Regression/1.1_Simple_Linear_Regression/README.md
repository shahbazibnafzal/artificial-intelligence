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
<img width="395" alt="image" src="https://github.com/user-attachments/assets/efe7b980-5eab-482f-bf02-41bd1a132486">

### Visualizing the test set result

```python
plt.scatter(X_test, Y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs Experience (test set)")
plt.ylabel("Salary")
plt.xlabel("Expeience")
plt.show()
```
<img width="397" alt="image" src="https://github.com/user-attachments/assets/b40ae7cc-a2f1-4dd9-8a60-b8c249645fa1">

## Model evaluation

Two common model evaluation techniques are:

**1. Train and test on the same dataset:**
Here we don't split the dataset while training the model, rather train on the entire dataset and the test on a portion of the data.

**2. Train/Test split:**
Here we split our dataset into training and test set. We train our model on the training set with labeled data and predict the test set and then compare the predicted values with the labels of test set.

**Training Accuracy**
Percentage of correct prediction made on the test set of the model on which the data has been trained on.
- Training and testing on the same dataset produces a hgh training accuracy but that's not necessarly a good thing.
- It will produce correct output for the data that it has already been trained on but might produce wrong output for new data which is called overfitting.

> Overfitting: If a model is overly trained to the dataset, it may capture noise and produce a non-generalized model.

**Out of sample accuracy**
Percentage of correct predictions model make on the data that the model has not been trained on.
- High out of sample accuracy is desired. 