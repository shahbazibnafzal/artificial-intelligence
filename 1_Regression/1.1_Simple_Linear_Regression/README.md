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

> Train test split is usually a good choice however it is hightly dependant on which dataset the model has trained and tested.

We can resolve this by using K-fold cross validation.

### K-fold cross validation

Basically we create mutiple folds of the dataset and then we iterate our process of splitting the training and testing dataset by assigning different fold in the testing set and train our model and then at the end we get the model that has been trained on different datapoints then the result is averaged to produce a more consistent out of sample accuracy.

## Accuracy Metrics

We learnt about ordinary least squared value where we calculate the minnimum sum of residual error and get the line of best fit.
These errors are calculated in three ways.

- Mean absolute error:
  This calculates the average of the absolute errors (without squaring) between observed and predicted values.

  ```math
    \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  ```

  Useful when you want a measure less sensitive to outliers, as it doesn’t square the errors. MAE provides a direct average distance between predictions and actuals in the original units of the data.

- Mean squared error
  This is the average of the squared residuals between observed and predicted values.

  ```math
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  ```
  Common in regression analysis, MSE penalizes larger errors more heavily than smaller ones (because of squaring), making it sensitive to outliers. It’s popular because it’s easy to compute and differentiable, which aids in optimization.

- Root mean squared error
  This is the square root of the MSE, providing a measure in the same units as the original data.
  
  ```math
  \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
  ``` 
  
  RMSE is useful when you want the error in the same units as the target variable, making it more interpretable. Like MSE, it’s sensitive to outliers, as errors are squared before averaging.
