# Step 3: Train

## Goal

Train a small model.

## Before you start

Finish Step 2.
Check `data/landmarks.csv`.
Keep a separate train set and test set.

The train set is the examples used to teach the model.
The test set is new examples used to check the model.
Accuracy on this dataset is too optimistic because pictures of the same person are in both sets.
A confusion matrix shows which signs the model mixes up.

We will also create an `idle` class.
It means no hand or a relaxed hand.

## What to do

1. Read the landmark data.
2. Split it into train and test sets.
3. Compare several small scikit-learn models.
4. Check accuracy and a confusion matrix.
5. Check the signs that are often mixed up.
6. Add the `idle` class to the plan.
7. Save the model as `model.joblib`.

Later, if the model is too slow on the Pi, we will look at TFLite.

## Commands

Run these commands later:

`python step3_train/compare_models.py`

This compares Logistic Regression, RBF SVM, Extra Trees, Random Forest,
KNN, and Histogram Gradient Boosting using accuracy, balanced accuracy, and
macro-F1. It saves each final model in `models/`, writes
`models/comparison_metrics.csv`, and copies the best careful-split model to
`step3_train/model.joblib`.

Test a saved model live with the webcam:

`python step3_train/test_live.py --model rbf_svm.joblib`

Use the same command with `extra_trees.joblib`, `random_forest.joblib`, or
another filename in `models/` to compare live behavior. Press `q` to quit.

## What you should see

You should see a training result, a test result, and a confusion matrix.
You should have `model.joblib`.

## If it does not work

Check that the CSV has enough rows.
Check that every label is spelled the same way.
Check that the train and test sets are not empty.
Write down signs that need more pictures.

## Checklist

- [ ] Train and test sets prepared
- [ ] Models compared
- [ ] Accuracy checked
- [ ] Confusion matrix checked
- [ ] `idle` class planned
- [ ] `model.joblib` saved
