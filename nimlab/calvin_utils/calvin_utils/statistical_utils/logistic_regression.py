import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut

class LogisticRegression():
    """
    This is a class which will run either a multinomial logistic or a logistic regression 
    depending upon the number of things you are trying to classify. 
    
    Params:
        outcome_matrix : pd.DataFrame
            - this is the dataframe with one-hot encoding. 
            - Observations are rows, classes are columns. 
            - One "1" per row
        design_matrix : pd.DataFrame
            - This is a design matrix for typical regression setups.
            
            
    Notes:
    fitting methods
    Newton-Raphson (newton): This is the default solver. It uses the score and Hessian functions and can be very fast for models that converge well. However, it might run into problems with models that are hard to optimize.

    Limited-memory Broyden Fletcher Goldfarb Shanno Algorithm (lbfgs): Default
        This is a good general-purpose solver that works well for a wide range of problems. 
        It's especially useful when dealing with large datasets or models because it uses an approximation to the 
        Hessian to guide its search without needing to store the entire Hessian matrix.

    Broyden Fletcher Goldfarb Shanno Algorithm (bfgs): 
        Similar to lbfgs, but it uses the full Hessian matrix. 
        It can be more accurate but is less memory efficient. 
        It's more suitable for smaller datasets or models.

    Conjugate Gradient (cg): 
        This solver is useful for problems with a large number of parameters. 
        It uses an iterative method to converge to the minimum, which can be more memory-efficient for large models.

    Newton-Conjugate Gradient (ncg): 
        This method uses a line search along conjugate directions, 
        which can be more efficient for some problems compared to the standard Newton-Raphson method.

    Powell s Method (powell): 
        This is a direction-set method that does not require the calculation of gradients, 
        which can be advantageous for certain types of problems, particularly those with discontinuous derivatives.
    """
    
    def __init__(self, outcome_matrix, design_matrix):
        self.outcome_matrix = outcome_matrix
        self.design_matrix = design_matrix
        
    def choose_method(self):
        if self.outcome_matrix.shape[1] > 1:
            self.multinomial_logistic()
        else:
            self.binomial_logistic()
    
    def multinomial_logistic(self):
        # Fit the regression model
        model = sm.MNLogit(self.outcome_matrix, self.design_matrix)
        self.results = model.fit()
        print("----INTERPRETATION KEY----")
        for i, cat in enumerate(self.outcome_matrix.columns):
            if i==0:
                print(f"reference_category : {cat}")
            else:
                print(f"y={i-1} : {cat}")
        
    def binomial_logistic(self): 
        model = sm.Logit(self.outcome_matrix, self.design_matrix)
        self.results = model.fit()
    
    @staticmethod
    def run_loocv(outcome_matrix, design_matrix):
        loo = LeaveOneOut()
        y_true = []
        y_pred = []
        
        for train_index, test_index in tqdm(loo.split(design_matrix)):
            X_train, X_test = design_matrix.iloc[train_index], design_matrix.iloc[test_index]
            y_train, y_test = outcome_matrix.iloc[train_index], outcome_matrix.iloc[test_index]
            
            if outcome_matrix.shape[1] > 1:
                model = sm.MNLogit(y_train, X_train)
            else:
                model = sm.Logit(y_train, X_train)
            results = model.fit(disp=0)  # disp=0 suppresses the output
            
            # Predict the test case
            test_pred = results.predict(X_test).to_numpy().argmax(1)
            y_true.append(y_test.values.argmax(1))
            y_pred.append(test_pred)
        
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Compute the metrics for each class
        accuracy = np.diag(cm).sum() / cm.sum()
        sensitivity = np.diag(cm) / np.sum(cm, axis=1)
        specificity = np.diag(cm) / np.sum(cm, axis=0)
        
        # Return the metrics
        return {
            'accuracy': accuracy,
            'class_wise_sensitivity': sensitivity,
            'overall_sensitivity': np.mean(sensitivity),
            'class_wise_specificity': specificity,
            'overall_specificity': np.mean(specificity),
            'confusion_matrix': cm
        }

    def run(self):
        self.choose_method()
        print(self.results.summary2)
        return self.results
