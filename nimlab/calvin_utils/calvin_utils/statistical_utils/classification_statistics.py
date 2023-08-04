from sklearn.metrics import accuracy_score, auc, roc_curve, accuracy_score
from calvin_utils.statistical_utils.distribution_statistics import bootstrap_distribution_statistics
import pandas as pd

def compute_accuracy(sample, threshold, y_true_variable, independent_variable):
    """
    Computes the accuracy for a given threshold.

    Parameters:
    - sample: DataFrame with the data.
    - threshold: float with the threshold to use for classifying the scores.
    - y_true_variable: string with the name of the column containing the true binary labels.
    - independent_variable: string with the name of the column containing the independent variable (classifier scores).

    Returns:
    - Scalar with the accuracy.
    """
    y_true = sample[y_true_variable]
    scores = sample[independent_variable]
    
    predictions = [0 if score <= threshold else 1 for score in scores]
    return accuracy_score(y_true, predictions)

def compute_auc(sample, y_true_variable, independent_variable):
    """
    Computes the Area Under the Curve (AUC) for the Receiver Operating Characteristic (ROC) curve.

    Parameters:
    - sample: DataFrame with the data.
    - y_true_variable: string with the name of the column containing the true binary labels.
    - independent_variable: string with the name of the column containing the independent variable (classifier scores).

    Returns:
    - Scalar with the AUC.
    """
    y_true = sample[y_true_variable]
    scores = sample[independent_variable]

    # calculate the false positive rate and true positive rate for all thresholds of the classification
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    # calculate the AUC and return it
    return auc(fpr, tpr)

def calculate_point_accuracy_and_distribution(data_df, scores, y_true, bootstrap_samples):
    # Calculate accuracies for all thresholds
    thresholds_unique = sorted(list(set(scores)))  # get unique score values and sort them
    accuracies = []

    bootstrap_results = {}
    for threshold in thresholds_unique:
        predictions = [0 if score <= threshold else 1 for score in scores]
        accuracy = accuracy_score(y_true, predictions)
        
        # Bootstrap Distribution
        func_args = {'threshold': threshold, 'y_true_variable': y_true, 'independent_variable': scores}
        bootstrap_results[f'{threshold}'] =  bootstrap_distribution_statistics(data_df, compute_accuracy, func_args, bootstrap_samples=bootstrap_samples)

    # Create a DataFrame to store thresholds and corresponding accuracies
    df_accuracies = pd.DataFrame({
        "Threshold": thresholds_unique,
        "Accuracy": accuracies
    })
    return df_accuracies, bootstrap_results

def compute_sensitivity_specificity(data, y_true_variable, independent_variable, threshold):
    y_true = data[y_true_variable]
    scores = data[independent_variable]

    # Compute predictions based on threshold
    predictions = [1 if score <= threshold else 0 for score in scores]

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()

    # Compute and return sensitivity and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity