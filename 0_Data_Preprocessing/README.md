# The Machine Learning Process

## 3 main steps in the journey of data.

1. Data Preprocessing

   - Import the data
   - Clean the data
   - Split data into training and test sets

2. Modelling

   - Build the model
   - Train the model
   - Make prediction

3. Evaluations
   - Calculate performance metrics
   - Make a verdict

[Google collab link](https://colab.research.google.com/)

## Importance of splitting the dataset into training and test set

When we have a labeled data (historical data with result), we can split the data into training and test set.
We train our model with the training set and make predection on the test set and then we compare the predicted results with the actual result of the test set to see how close our model is predicting.

## Independant and dependant variables.

**Independant variable (X):**
The parameters (columns) of the records based on which the result is dependant are called as independant variables or features and are denoted with X.

**Dependant variable (Y):**
The value we want to predict based on different feaures or values are called dependant or target variables.

## Feature Scaling

Feature scaling is applied to independant variables (columns)

There are two common fetaure scaling techniques:

1. Normaliation

```math
X' = (X - X min) / (X max - X min)
```

Each value in the column is calculated this way, so the values will be between 0 and 1.

2. Standardization

```math
X' = (X - mean) / standard deviation
```

Each value in the column is calculated this way, so the values will be between -3 and 3 (Except outliers).

## Data preprocessing template

There are a few steps that we always need to perform for pre process the data and make them ready for modelling.

1. Importing the libraries to use.
2. Importing the datasets and defining the independant and dependant variables.
3. Taking care of missing data.
4. Encoding the categorical values (Converting the labels to numeric values to perform the calculations on them).
5. Splitting the data into training and testing set.
6. Feature scaling (Not used all the time but based on the dataset it can be required).

### Importing the libraries

Let's see how to import libraries by importing some common livraries used for preprocessing and plotting the data.

```python
import numpy as np # Used to perform array operations.
import pandas as pd # Used to import and perform actions on the datasets.
import matplotlib.pyplot as plt # Used to plot charts for data vizualization.
```

### Importing the dataset and defining the dependant and independant variables

```python
dataset = pd.read_csv("Data.csv") # Reading the csv file with pandas and storing it as a dataframe
print(dataset) # This will print the dataset
X = dataset.iloc[:, :-1].values # Storing all the values of independant variables
Y = dataset.iloc[:, -1].values # Storing all the values of the dependant or target variable
```

### Identifying the missing data

```python
dataset.isnull() # This will create a table with true and false value for the dataset. true indicates the missing data
dataset.isnull().sum() # This will give the number of missing values in each column.
```

### Taking care of missing data

There are several techniques of taking care of missing data in a dataset.

1. Remove the records (If the dataset is huge and there are only a few records with missing data).
2. Replace the missing value with the average (mean) / median / mode value.

```python
from sklearn.impute import SimpleImputer # Simple Imputer has built in functions to handle missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # Create an instance of simple imputer
imputer.fit(X[:, 1:3]) # Fit the imputer to only numerical values, don't include categorical values
X[:, 1:3] = imputer.transform(X[:, 1:3]) # Transform and replace the values with the new values
```

### Encoding the categorical values

Categorical values need to be convered to numeric values however we can't just assign some numeric values that has order.
We can use OneHotEncoder to create columns for each category and create binary values 0 or 1 for each column.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
```
