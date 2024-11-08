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

### Importing the dataset and defining the dependent and independant variables

```python
dataset = pd.read_csv("Data.csv") # Reading the csv file with pandas and storing it as a dataframe
print(dataset) # This will print the dataset
X = dataset.iloc[:, :-1].values # Storing all the values of independant variables
Y = dataset.iloc[:, -1].values # Storing all the values of the dependant or target variable
```