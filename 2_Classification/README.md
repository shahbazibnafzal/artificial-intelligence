# Classification

- A supervised approach to to classify the data into a category.
- The target variable is a categorical variable.

## Application

- Predict which category a customer belong to.
- Predict whether a customer switches to another provider?
- Predict whether a customer will respond to an ad campaign.
- Email filtering
- Face recognition

## Evaluation metrics in classification



We train the data on training set and then make prediction on the test data.

So we have predicted and actual value of the test set. We can compare those to evaluate the accuracy of the model.

There 3 common evaluation strategies.

**1. Jaccord index:**
y: Actual labels
\hat{y}: Predicted labels

Jacob index is the size of the intersection of the actual and predicted values divided by union of actual and predicted values.

```math
J(y, \hat{y}) = \frac{|A \cap \hat{y}|}{|A \cup \hat{y}|}
```

**2. Confusion matrix (F1-score):**

![confusion matrix](https://github.com/user-attachments/assets/bfec8e2b-6b2e-40af-b728-caf2357eace3)

We can create a confusion marix with actual and predicted value.

TP = True positive: Correctly predicted the true value for a binary classification.
TN = True negative: Correctly predicted the false value for a binary classification.
FP = False positive: Incorrectly predicted the true value for a binary classification.
FN = False negative: Incorrectly predicted the false value for a binary classification.

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 score = 2 * (precision * recall) / (precision + recall)

F1: 1 is best, 0 is worst.

**3. Log loss:**
