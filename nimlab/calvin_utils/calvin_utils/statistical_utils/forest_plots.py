#Imports for Forest Plot
import os
import numpy as np
import pandas as pd
import forestplot as fp
import statsmodels.api as sm
import statsmodels.formula.api as smf

class ForestPlot():
    '''
    This is a class which takes a statsmodels model object (from a fitted regression model)
    and generates a forest plot of the regession coefficients for easy reading. 
    
    Parameters:
    model : RegressionResults
        The fitted regression model model from statsmodels.
    table : Boolean
        The True/False Boolean of whether you would like to present to forest plot as a table.
    sig_difits : int
        The number of significant digits to display. Defaults to 2. 
    '''
    def __init__(self, model, sig_digits=2, out_dir=None, table=False):
        self.model=model
        self.sig_digits=sig_digits
        self.out_dir=out_dir
        self.table=table

    def table_prep(self):
        '''
        If we want to use a table with less than 6 regressors, the table output will be malformed. 
        To address this, we fill the bottom of the self.data_for_plot dataframe with np.NaN 
        to expand over 6 rows.
        '''
        if self.table and len(self.data_for_plot) < 7:
            additional_rows_needed = 7 - len(self.data_for_plot)
            
            # Create a DataFrame with the additional rows filled with np.NaN
            additional_rows = pd.DataFrame({
                'estimate': [np.NaN] * additional_rows_needed,
                'll': [np.NaN] * additional_rows_needed,
                'hl': [np.NaN] * additional_rows_needed,
                'label': [''] * additional_rows_needed,  # Assuming label can remain as an empty string
                'pvalue': [np.NaN] * additional_rows_needed,
                'group': [''] * additional_rows_needed
            })

            # Append the additional rows to the data_for_plot DataFrame
            self.data_for_plot = pd.concat([self.data_for_plot, additional_rows], ignore_index=True)

    def prepare_results(self):
        '''
        Method to extract results from a model and prepare a dataframe for forest plot generation
        '''
        # Extracting coefficients, confidence intervals, and p-values
        params = self.model.params
        conf = self.model.conf_int()
        pvalues = self.model.pvalues
        variables = self.model.params.index
        group = ['Coefficient'[::]] * len(variables)

        # Preparing the data for the forest plot
        self.data_for_plot = pd.DataFrame({
            'estimate': params,
            'll': conf.iloc[:, 0],
            'hl': conf.iloc[:, 1],
            'label': variables,
            'pvalue': pvalues,
            'group': group
        })
        self.data_for_plot.reset_index(inplace=True, drop=True)

    def create_and_display_forest_plot(self, x_label="Regression coefficient", odds_ratio=False, flush=False):
        """
        Generate and display a forest plot from the outputs of a regression model using forestplot.py.
        """
        # Generate the forest plot
        ax = fp.forestplot(dataframe=self.data_for_plot,
                #neccessary inputs
                estimate="estimate",  # col containing estimated effect size 
                varlabel="label",  # column containing variable label
                
                #Additional Plotting Inputs
                ll="ll", hl="hl",  # columns containing conf. int. lower and higher limits
                pval="pvalue" if not odds_ratio else None,  # Column of p-value to be reported on right

                # Axis Labels
                xlabel=x_label,
                ylabel='Est.(95% Conf. Int.)',
                
                # Forest Plot Configuration
                decimal_precision=self.sig_digits,
                capitalize='capitalize',
                color_alt_rows=False,  # Gray alternate rows
                table=self.table,
                flush=flush,
                
                #   # Image Configuration
                **{"marker": "D",  # set maker symbol as diamond
                    "markersize": 150,  # adjust marker size
                    "xlinestyle": (0, (10, 5)),  # long dash for x-reference line 
                    "xlinecolor": "#808080",  # gray color for x-reference line
                    "xtick_size": 12,  # adjust x-ticker fontsize
                    'fontfamily': 'helvetica'
                    }  
            )
        self.fig = ax.figure
        self.fig.show()
        
    def figure_saver(self, title="regression_forest_plot"):
        '''
        Method to save the forest plot.
        '''
        # Save the plot as PNG and SVG
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            self.fig.savefig(os.path.join(self.out_dir, f"{title}.png"), bbox_inches='tight')
            self.fig.savefig(os.path.join(self.out_dir, f"{title}.svg"), bbox_inches='tight')
            print(f'Saved to {self.out_dir} as {title}.svg and .png')
            
    def run(self):
        '''
        Orchestrator method.
        '''
        self.prepare_results()
        self.table_prep()
        self.create_and_display_forest_plot()
        self.figure_saver()

class OddsRatioForestPlot(ForestPlot):
    """
    This class inherits from the ForestPlot class and is designed to accept a pandas DataFrame
    containing categorical data for a single outcome and predictor. It runs logistic regression
    for each category of the predictor variable and stores the coefficients, p-values, and
    confidence intervals. It then exponentiates the coefficients and their confidence intervals
    to obtain odds ratios and generates a forest plot to visualize the results.

    Parameters:
    data : pandas.DataFrame
        DataFrame containing the study data with columns for outcome, predictor, and category.
    outcome_col : str
        Name of the column in the DataFrame containing the outcome variable.
    predictor_col : str
        Name of the column in the DataFrame containing the predictor variable.
    category_col : str
        Name of the column in the DataFrame containing the category variable.
    treatment : str, optional
        This is a string which defines the STRING in the treatment class. It will make sure rows matching the string are set to 1. 
        This is critical for enforcing the sign of the odds ratio. If left as None, will arbitrarily assign 1/0 to rows. 
    covariates : list, optional
        List of column names in the DataFrame to be included as covariates in the logistic regression (default: None).
    sig_digits : int, optional
        Number of significant digits to display in the forest plot (default: 2).
    out_dir : str, optional
        Directory path where the plot should be saved (default: None).
    table : bool, optional
        Flag to indicate whether the plot should be displayed as a table (default: False).
    log_odds : bool, optional
        Flag to indicate whether to display log odds or not. 
    """

    def __init__(self, data, outcome_col, predictor_col, category_col, treatment = None, covariates=None, sig_digits=2, out_dir=None, table=False, log_odds=False):
        self.data = data
        self.outcome_col = outcome_col
        self.predictor_col = predictor_col
        self.predictor_treatment = treatment
        self.category_col = category_col
        self.covariates = covariates if covariates else []
        self.sig_digits = sig_digits
        self.out_dir = out_dir
        self.table = table
        self.log_odds = log_odds
        if self.log_odds==True:
            self.x_label = "Log Odds Ratio"
        else:
            self.x_label = "Odds Ratio"
        
    def create_dummy_variables(self):
        """
        Create label-encoded variables for all columns except the category columns.
        """
        # Identify columns to encode (exclude 'category' and non-string columns)
        columns_to_encode = [col for col in self.data.columns if col not in [self.category_col] and self.data[col].dtype == 'object']
        
        # Iterate over each column and apply label encoding
        for col in columns_to_encode:
            if col == self.predictor_col:
                if self.predictor_treatment is not None: 
                    self.data[col] = np.where(self.data[col]==self.predictor_treatment, 1, 0)
                    continue 
            self.data[col] = pd.factorize(self.data[col])[0]

    def construct_formula(self):
        """
        Construct the formula for the logistic regression using patsy.
        """
        self.formula = f"{self.outcome_col} ~ {self.predictor_col}"
        if self.covariates:
            self.formula += " + " + " + ".join(self.covariates)
        print("Running formula: ", self.formula)
    
    def run_logistic_regression(self, category=None, debug=False):
        """
        Run logistic regression for a specific category of the predictor variable.
        """
        if category is not None:
            data_subset = self.data[self.data[self.category_col] == category]
        else:
            data_subset = self.data
        if debug:
            print("~~Col: ", category)
            print(data_subset[self.predictor_col])
            print(data_subset[self.outcome_col])
        try:
            logit_model = smf.logit(self.formula, data=data_subset).fit()
        except sm.tools.sm_exceptions.PerfectSeparationError as e:
            backup_formula = self.formula + " - 1"
            logit_model = smf.logit(backup_formula, data=data_subset).fit()
            print(F"Perfect separation error detected. \n RESULTS FOR {category} UNRELIABLE")
        return logit_model
    
    def get_coef_and_ci(self, logit_model):
        coef = logit_model.params[self.predictor_col]
        pvalue = logit_model.pvalues[self.predictor_col]
        conf_int = logit_model.conf_int().loc[self.predictor_col, :]
        
        if self.log_odds==True:
            odds_ratio = coef
            ci_low, ci_high = conf_int
        else:
            odds_ratio = np.exp(coef)
            ci_low, ci_high = np.exp(conf_int)
            
        return odds_ratio, ci_low, ci_high, pvalue
        
    
    def calculate_odds_ratios(self):
        """
        Calculate odds ratios and their confidence intervals for each category of the predictor.
        """
        # Generate Each Group's Odds Ratio
        results = []
        for category in self.data[self.category_col].unique():
            logit_model = self.run_logistic_regression(category)
            
            odds_ratio, ci_low, ci_high, pval = self.get_coef_and_ci(logit_model)

            results.append((category, odds_ratio, ci_low, ci_high, pval))
        
        # Generate the Summary Odds Ratio
        logit_model = self.run_logistic_regression()
        odds_ratio, ci_low, ci_high, pval = self.get_coef_and_ci(logit_model)
        results.append(('Summary', odds_ratio, ci_low, ci_high, pval))

        self.data_for_plot = pd.DataFrame(results, columns=['label', 'estimate', 'll', 'hl', 'pvalue'])
        
    def run(self):
        """
        Orchestrator method to calculate odds ratios and generate the forest plot.
        """
        self.create_dummy_variables()
        self.construct_formula()
        self.calculate_odds_ratios()
        # self.calculate_overall_odds()
        super().table_prep()
        super().create_and_display_forest_plot(x_label=self.x_label, odds_ratio=True, flush=True)
        super().figure_saver(title="log_odds_ratio_forest_plot" if self.log_odds else "odds_ratio_forest_plot")