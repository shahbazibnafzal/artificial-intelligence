# K - Nearest Neighbors

- A method for classifying data points based on their similarity to other cases.
- Cases that are near each other are said to be neighbors.
- Based on similar cases with same class labels are near each other.
- Euclidean distance is a common metric used to measure how "close" or "similar" two points are in the feature space.

```math
\text{distance}(A, B) = \sqrt{(A_1 - B_1)^2 + (A_2 - B_2)^2 + \dots + (A_n - B_n)^2}
```

# How does the KNN algorithm works?

1. Pick a value for K (e.g. 5)
2. Calculate the distance of unknown case from all cases.
3. Select the K-observaions in the training data that are nearest to the unknown data points.
4. Predict the response of the unknown data point using the most popular response value from K-nearest neighbors.

## How to choose the best K-value?

- If we choose a very low K value (K=1), it can be a bad prection because what if the nearest point was an anomaly.
- Low value of K causes a complex model, and cause overfitting.
- The model is not generalized enough to predict on out of sample datapoint.
- If we use a high value of K, the data will be overly generalized.
- So, best way would be to try different values of K in your model and check the acuuracy of the model.

> K-NN can also be used for regression 

### How do we predict the continuous value with K-NN?

We find the K nearest neighbors (i.e. 3 nearest datapoints) and then take the median of the target values of those 3 datapoints.

