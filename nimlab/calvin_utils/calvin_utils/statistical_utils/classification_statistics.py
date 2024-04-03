from sklearn.metrics import accuracy_score, auc, roc_curve, accuracy_score
from calvin_utils.statistical_utils.distribution_statistics import bootstrap_distribution_statistics
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

class ClassificationEvaluation:
    """
    This is a class which will either take a fitted Statsmodels Model Object and a dataframe of observations,
    or it will take a dataframe of predictions and a dataframe of observations. 
    
    Will extract various metrics such as accuracy, sensitivity, specificity, PPV, and NPV.
    Will generate a heatmap, normalized as you please.
    
    Notes:
    1. No Normalization:
        Description: The confusion matrix contains the raw counts of predictions for each combination of true and predicted classes.
        When to Use: When you're interested in the absolute numbers of observations for each prediction outcome.
    2. Normalization by True Classes ('true'):
        Description: Each element in the confusion matrix is divided by the sum of elements in its corresponding true class row. This results in each row summing to 1.
        Interpretation: The values represent the proportion of the actual (true) class instances that were predicted as each class. It's useful for understanding the distribution of predictions for each true class.
        When to Use: To analyze the classifier's ability to correctly identify each true class, especially when the distribution of classes is imbalanced.
    3. Normalization by Predicted Classes ('pred'):
        Description: Each element in the confusion matrix is divided by the sum of elements in its corresponding predicted class column. This results in each column summing to 1.
        Interpretation: The values indicate the proportion of predictions for each class that were actually instances of the true classes. It helps in assessing the precision or positive predictive value of predictions for each class.
        When to Use: To evaluate how confidently the model predicts each class, particularly when some classes are prone to being overpredicted.
    4. Normalization by All Elements ('all'):
        Description: Each element in the confusion matrix is divided by the total number of observations. This results in the entire matrix summing to 1.
        Interpretation: The values show the proportion of all observations that fall into each combination of true and predicted classes. It provides a holistic view of the classifier's performance across all classes.
        When to Use: When you want a high-level overview of the model's performance, including both the rate of correct predictions and the distribution of errors, relative to the entire dataset.
    
    Choosing a Normalization Method:
    The choice of normalization method depends on what 
    aspect of the model's performance you're most interested in:

    Use no normalization for a straightforward count of each outcome.
    Normalize by true classes ('true') to focus on sensitivity, recall, or the classifier's ability to identify each class.
    Normalize by predicted classes ('pred') to concentrate on precision or the likelihood that predicted instances of each class are correct.
    Normalize by all elements ('all') for a comprehensive view of the model's performance relative to the total number of observations.
    """
    
    def __init__(self, fitted_model, observation_df, normalization=None, predictions_df=None):
        """
        Initializes the ModelEvaluation with model results and the true outcomes.
        
        Args:
            fitted_model: The result object from a fitted statsmodels MNLogit model.
            outcome_matrix: A pandas DataFrame with the true class outcomes in one-hot encoded format.
            normalization: Normalization method for the confusion matrix (None, 'true', 'pred', 'all').
            predictions_df: Manually entered DataFrame of predictions, can contain probabilities or dummy-coded predictions.
        """
        self.results = fitted_model
        self.outcome_matrix = observation_df
        self.normalization = normalization
        self.predictions_df = predictions_df
        
    def get_predictions(self):
        """
        Takes a model or a DF of probabilities (or dummy-coded predictions) and returns the prediction. 
        """
        if self.predictions_df is not None:
            self.raw_predictions = self.predictions_df.to_numpy()
        else:
            self.raw_predictions = self.results.predict()
        self.predictions = self.raw_predictions.argmax(1)
    
    def get_observations(self):
        """
        Takes a DF of dummy-coded observations and 
        """
        self.raw_observations = self.outcome_matrix.to_numpy()
        self.observations = self.raw_observations.argmax(1) 
    def calculate_confusion_matrix(self):
        """Calculates the confusion matrix from predictions and observations."""
        self.conf_matrix = confusion_matrix(self.observations, self.predictions, normalize=self.normalization)
    
    def extract_confusion_components(self):
        """Extracts True Positive, True Negative, False Positive, and False Negative counts from the confusion matrix."""
        self.TP = self.conf_matrix[1, 1]
        self.TN = self.conf_matrix[0, 0]
        self.FP = self.conf_matrix[0, 1]
        self.FN = self.conf_matrix[1, 0]
    
    def calculate_metrics(self):
        """Calculates accuracy, sensitivity, specificity, PPV, and NPV based on the confusion matrix components."""
        self.accuracy = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        self.sensitivity = self.TP / (self.TP + self.FN)
        self.specificity = self.TN / (self.TN + self.FP)
        self.PPV = self.TP / (self.TP + self.FP)
        self.NPV = self.TN / (self.TN + self.FN)
    
    def display_metrics(self):
        """Prints the calculated evaluation metrics."""
        print("Accuracy:", self.accuracy)
        print("Sensitivity:", self.sensitivity)
        print("Specificity:", self.specificity)
        print("PPV:", self.PPV)
        print("NPV:", self.NPV)
        
    def plot_confusion_matrix(self):
        """Plots a heatmap of the confusion matrix."""
        labels = np.unique(self.outcome_matrix.columns)

        plt.figure(figsize=(8, 6))
        sns.heatmap(self.conf_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Normalized Confusion Matrix" if self.normalization else "Confusion Matrix")
        plt.show()
    
    def run(self):
        """Orchestrates the calculation and display of all evaluation metrics."""
        self.get_predictions()
        self.get_observations()
        self.calculate_confusion_matrix()
        self.extract_confusion_components()
        self.calculate_metrics()
        self.display_metrics()
        self.plot_confusion_matrix()

class MulticlassOneVsAllROC(ClassificationEvaluation):
    """
    Extends ModelEvaluation to include ROC curve generation for each class
    in a multinomial logistic regression model.
    """
    def plot_confusion_matrix(self):
        pass
    
    def plot_roc_curves(self):
        """
        Plots ROC curves for each class using a One-vs-Rest approach.
        """
        n_classes = self.outcome_matrix.shape[1]
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Binarize the output
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.raw_observations[:, i], self.raw_predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot all ROC curves
        plt.figure()
        
        colors = iter(sns.color_palette("hsv", n_classes))
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('One-vs-Rest ROC Curves')
        plt.legend(loc="lower right")
        plt.show()

    def run(self):
        """
        Orchestrate the evaluation including confusion matrix, metrics, and ROC curves.
        """
        super().run()
        self.plot_roc_curves()

class MacroAverageROC(MulticlassOneVsAllROC):
    """
    Extends MulticlassModelEvaluation to include the generation of a macro-average ROC curve
    for a multinomial logistic regression model.
    """
    
    def plot_macro_average_roc_curve(self):
        """
        Plots a macro-average ROC curve for the multinomial logistic regression model.
        
        This is like a meta-analytic ROC, but it averages all ROCs together with equal weight.
        """
        n_classes = self.outcome_matrix.shape[1]
        # Compute ROC curve and ROC area for each class
        all_fpr = np.unique(np.concatenate([np.linspace(0, 1, 100) for _ in range(n_classes)]))
        
        # Then interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(self.outcome_matrix.iloc[:, i], self.results.predict()[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        
        fpr = all_fpr
        tpr = mean_tpr
        roc_auc = auc(fpr, tpr)
        
        # Plot the macro-average ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='Macro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc), color='navy', linestyle=':', linewidth=4)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Macro-average ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    def run(self):
        """
        Orchestrates the evaluation including the macro-average ROC curve.
        """
        super().run()
        self.plot_macro_average_roc_curve()
        
class MicroAverageROC(MacroAverageROC):
    """
    Extends MacroAverageROC to include the generation of both macro-average and micro-average ROC curves
    for a multinomial logistic regression model.
    
    This is like a meta-analytic ROC. It averages the ROC curves, but with respect to sample weight. 
    """
    
    def plot_micro_average_roc_curve(self):
        """
        Plots a micro-average ROC curve for the multinomial logistic regression model.
        """
        n_classes = self.outcome_matrix.shape[1]
        # Aggregate all false positive rates and true positive rates
        fpr, tpr, thresholds = roc_curve(self.outcome_matrix.to_numpy().ravel(), self.results.predict().ravel())
        
        roc_auc = auc(fpr, tpr)
        
        # Plot the micro-average ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='Micro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc), color='deeppink', linestyle=':', linewidth=4)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Micro-average ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    def run(self):
        """
        Orchestrates the evaluation including both the macro-average and micro-average ROC curves.
        """
        super().run()
        # self.plot_macro_average_roc_curve()
        self.plot_micro_average_roc_curve()

class ComprehensiveMulticlassROC(MicroAverageROC):
    """
    Extends ClassificationEvaluation to include ROC curve generation for a multinomial logistic regression model.
    
    ROC Curves Included:
    - First Curve: Individual ROC curves for the classification of each class against all other classes. This is 
      known as the One-vs-Rest (OvR) approach, where each class is considered as the positive class once, and 
      all other classes are combined to form the negative class, producing a ROC curve for each class.
      
    - Second Curve (Macro Average): A single ROC curve that represents the average performance across all classes. 
      The True Positive Rate (TPR) and False Positive Rate (FPR) are averaged for each class, treating each class 
      equally, regardless of its size or frequency in the dataset. This curve provides an overall performance measure 
      that gives equal weight to each class.
      
    - Third Curve (Micro Average): A single ROC curve that aggregates the contributions of all classes to compute the 
      overall TPR and FPR. This approach gives equal weight to each instance, summing the individual true positives, 
      false positives, true negatives, and false negatives across all classes before calculating TPR and FPR. The 
      micro-average ROC curve is especially useful in datasets with class imbalance, as it reflects the model's 
      performance across all instances, but biases towards more highly represented classes.
    """
    def run(self):
        """
        Orchestrates the evaluation including both the macro-average and micro-average ROC curves.
        """
        super().run()
                

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