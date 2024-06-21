import os
import re
import patsy 
import numpy as np
import statsmodels
import seaborn as sns
from typing import Type
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.tools.tools import maybe_unwrap_results
from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Imports for Forest Plot
import numpy as np
import forestplot as fp
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# For PDD
import statsmodels.discrete.discrete_model as smd


style_talk = 'seaborn-talk'    #refer to plt.style.available

def calculate_vif(df):

    """
    Calculates the variance inflation factor (VIF) for each feature in a pandas DataFrame.

    Parameters:
        df (pandas DataFrame): The DataFrame containing the features.

    Returns:
        vif (pandas DataFrame): A DataFrame containing the VIF values for each feature.
    """
    # Prepare features
    X = df.iloc[:, :]

    # Calculate VIF for each feature
    vif = pd.DataFrame()
    vif["features"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif


class Linear_Reg_Diagnostic():
    # base code


    """
    Diagnostic plots to identify potential problems in a linear regression fit.
    Mainly,
        a. non-linearity of data
        b. Correlation of error terms
        c. non-constant variance
        d. outliers
        e. high-leverage points
        f. collinearity

    Author:
        Prajwal Kafle (p33ajkafle@gmail.com, where 3 = r)
        Does not come with any sort of warranty.
        Please test the code one your end before using.
    """

    def __init__(self,
                 results: Type[statsmodels.regression.linear_model.RegressionResultsWrapper]) -> None:
        """
        For a linear regression model, generates following diagnostic plots:

        a. residual
        b. qq
        c. scale location and
        d. leverage

        and a table

        e. vif

        Args:
            results (Type[statsmodels.regression.linear_model.RegressionResultsWrapper]):
                must be instance of statsmodels.regression.linear_model object

        Raises:
            TypeError: if instance does not belong to above object

        Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> import statsmodels.formula.api as smf
        >>> x = np.linspace(-np.pi, np.pi, 100)
        >>> y = 3*x + 8 + np.random.normal(0,1, 100)
        >>> df = pd.DataFrame({'x':x, 'y':y})
        >>> res = smf.ols(formula= "y ~ x", data=df).fit()
        >>> cls = Linear_Reg_Diagnostic(res)
        >>> cls(plot_context="seaborn-paper")

        In case you do not need all plots you can also independently make an individual plot/table
        in following ways

        >>> cls = Linear_Reg_Diagnostic(res)
        >>> cls.residual_plot()
        >>> cls.qq_plot()
        >>> cls.scale_location_plot()
        >>> cls.leverage_plot()
        >>> cls.vif_table()
        """

        if isinstance(results, statsmodels.regression.linear_model.RegressionResultsWrapper) is False:
            raise TypeError("result must be instance of statsmodels.regression.linear_model.RegressionResultsWrapper object")

        self.results = maybe_unwrap_results(results)

        self.y_true = self.results.model.endog
        self.y_predict = self.results.fittedvalues
        self.xvar = self.results.model.exog
        self.xvar_names = self.results.model.exog_names

        self.residual = np.array(self.results.resid)
        influence = self.results.get_influence()
        self.residual_norm = influence.resid_studentized_internal
        self.leverage = influence.hat_matrix_diag
        self.cooks_distance = influence.cooks_distance[0]
        self.nparams = len(self.results.params)

    def __call__(self, plot_context='seaborn-paper'):
        # print(plt.style.available)
        with plt.style.context(plot_context):
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
            self.residual_plot(ax=ax[0,0])
            self.qq_plot(ax=ax[0,1])
            self.scale_location_plot(ax=ax[1,0])
            self.leverage_plot(ax=ax[1,1])
            plt.show()

        self.vif_table()
        return fig, ax


    def residual_plot(self, ax=None):
        """
        Residual vs Fitted Plot

        Graphical tool to identify non-linearity.
        (Roughly) Horizontal red line is an indicator that the residual has a linear pattern
        """
        if ax is None:
            fig, ax = plt.subplots()

        sns.residplot(
            x=self.y_predict,
            y=self.residual,
            lowess=True,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        residual_abs = np.abs(self.residual)
        abs_resid = np.flip(np.sort(residual_abs))
        abs_resid_top_3 = abs_resid[:3]
        for i, _ in enumerate(abs_resid_top_3):
            ax.annotate(
                i,
                xy=(self.y_predict[i], self.residual[i]),
                color='C3')

        ax.set_title('Residuals vs Fitted', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')
        return ax

    def qq_plot(self, ax=None):
        """
        Standarized Residual vs Theoretical Quantile plot

        Used to visually check if residuals are normally distributed.
        Points spread along the diagonal line will suggest so.
        """
        if ax is None:
            fig, ax = plt.subplots()

        QQ = ProbPlot(self.residual_norm)
        QQ.qqplot(line='45', alpha=0.5, lw=1, ax=ax)

        # annotations
        abs_norm_resid = np.flip(np.argsort(np.abs(self.residual_norm)), 0)
        abs_norm_resid_top_3 = abs_norm_resid[:3]
        for r, i in enumerate(abs_norm_resid_top_3):
            ax.annotate(
                i,
                xy=(np.flip(QQ.theoretical_quantiles, 0)[r], self.residual_norm[i]),
                ha='right', color='C3')

        ax.set_title('Normal Q-Q', fontweight="bold")
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Standardized Residuals')
        return ax

    def scale_location_plot(self, ax=None):
        """
        Sqrt(Standarized Residual) vs Fitted values plot

        Used to check homoscedasticity of the residuals.
        Horizontal line will suggest so.
        """
        if ax is None:
            fig, ax = plt.subplots()

        residual_norm_abs_sqrt = np.sqrt(np.abs(self.residual_norm))

        ax.scatter(self.y_predict, residual_norm_abs_sqrt, alpha=0.5);
        sns.regplot(
            x=self.y_predict,
            y=residual_norm_abs_sqrt,
            scatter=False, ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        abs_sq_norm_resid = np.flip(np.argsort(residual_norm_abs_sqrt), 0)
        abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
        for i in abs_sq_norm_resid_top_3:
            ax.annotate(
                i,
                xy=(self.y_predict[i], residual_norm_abs_sqrt[i]),
                color='C3')
        ax.set_title('Scale-Location', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel(r'$\sqrt{|\mathrm{Standardized\ Residuals}|}$');
        return ax

    def leverage_plot(self, ax=None):
        """
        Residual vs Leverage plot

        Points falling outside Cook's distance curves are considered observation that can sway the fit
        aka are influential.
        Good to have none outside the curves.
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(
            self.leverage,
            self.residual_norm,
            alpha=0.5);

        sns.regplot(
            x=self.leverage,
            y=self.residual_norm,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        leverage_top_3 = np.flip(np.argsort(self.cooks_distance), 0)[:3]
        for i in leverage_top_3:
            ax.annotate(
                i,
                xy=(self.leverage[i], self.residual_norm[i]),
                color = 'C3')

        xtemp, ytemp = self.__cooks_dist_line(0.5) # 0.5 line
        ax.plot(xtemp, ytemp, label="Cook's distance", lw=1, ls='--', color='red')
        xtemp, ytemp = self.__cooks_dist_line(1) # 1 line
        ax.plot(xtemp, ytemp, lw=1, ls='--', color='red')

        ax.set_xlim(0, max(self.leverage)+0.01)
        ax.set_title('Residuals vs Leverage', fontweight="bold")
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Standardized Residuals')
        ax.legend(loc='upper right')
        return ax

    def vif_table(self):
        """
        VIF table

        VIF, the variance inflation factor, is a measure of multicollinearity.
        VIF > 5 for a variable indicates that it is highly collinear with the
        other input variables.
        """
        vif_df = pd.DataFrame()
        vif_df["Features"] = self.xvar_names
        vif_df["VIF Factor"] = [variance_inflation_factor(self.xvar, i) for i in range(self.xvar.shape[1])]

        print(vif_df
                .sort_values("VIF Factor")
                .round(2))


    def __cooks_dist_line(self, factor):
        """
        Helper function for plotting Cook's distance curves
        """
        p = self.nparams
        formula = lambda x: np.sqrt((factor * p * (1 - x)) / x)
        x = np.linspace(0.001, max(self.leverage), 50)
        y = formula(x)
        return x, y
    
def model_diagnostics(optimal_model):

    """
    Plots a QQ plot, a residual plot, and a predicted vs observed plot for a given linear regression model.

    Parameters:
        optimal_model (statsmodels regression model): The optimal linear regression model.

    Returns:
        None
    """
    
    # Set color palette
    sns.set_style('white')
    sns.set_palette('Greys', 1, desat=1)
    
    # Calculate predicted values and residuals
    y_pred = optimal_model.predict()
    residuals = optimal_model.resid
    
    # Create plots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # QQ plot of residuals
    sm.qqplot(residuals, ax=axs[0], line='45')
    axs[0].lines[0].set_color('black')
    axs[0].set_title("QQ Plot of Residuals")

    # Residual plot
    axs[1].scatter(y_pred, residuals)
    axs[1].set_xlabel("Predicted values")
    axs[1].set_ylabel("Residuals")
    axs[1].set_title("Residual Plot")

    # Predicted vs observed plot
    axs[2].scatter(optimal_model.fittedvalues, optimal_model.model.endog)
    axs[2].plot(optimal_model.model.endog, optimal_model.model.endog, color='r')
    axs[2].set_xlabel("Predicted values")
    axs[2].set_ylabel("Observed values")
    axs[2].set_title("Predicted vs Observed Plot")


class EMMPlot():
    """
    This is a class which will take a formula, a dataframe, and a fitted model. 
    It will calculate the mean of the predictors, and then hold them all constant while allowing one to vary linearly over its range. 
    It will then make predictions across this range, and we will then be able to see the isolated relationship between varied predictor and prediction. 
    
    EMMs refer to the means for each level of a categorical variable adjusted for other variables in the model. 
    They are essentially adjusted means for groups after accounting for covariates in the model.
    
    NOTE
    There are 2 ways of generating EMMs, and will generate slightly different results
        - See this manuscript, which advocates for the 'observed value approach': https://gvpt.umd.edu/sites/gvpt.umd.edu/files/pubs/Hanmer%20and%20Kalkan%20AJPS%20Behind%20the%20Curve.pdf
    1) To use the 'observed values approach' is to make predictions on the outcomes based on the actual data, then average the preidctions
    2) To use the 'average case approach' is to average the actual data together, then make a prediction on it. 
        This code uses the 'average case approach'.
        Th
    """
    def __init__(self, formula, data_df, model):
        self.formula = formula
        self.data_df = data_df
        self.model = model
        
    def extract_unique_variables(self):
        # Remove the response variable part of the formula
        _, predictors = self.formula.split('~')
        # Identify categorical variables with 'C()' and mark them
        cat_vars = re.findall(r'C\((.*?)\)', predictors)
        cat_vars_dict = {var: 'categorical' for var in cat_vars}
        # Remove the 'C()' from the categorical variables
        predictors_no_cat = re.sub(r'C\((.*?)\)', r'\1', predictors)
        # Split based on '+' and ':' to separate all variables including those in interactions
        variables = re.split(r'\+|\*|\:|\-', predictors_no_cat)
        # Remove any white spaces and strip each variable name
        variables = [var.strip() for var in variables]
        # Remove duplicates while preserving order by converting to a dict and back to list
        unique_vars = list(dict.fromkeys(variables))
        # Remove empty strings that may have resulted from stripping
        unique_vars = [var for var in unique_vars if var]
        # Mark the remaining variables as continuous
        unique_vars_dict = {var: 'continuous' for var in unique_vars if var not in cat_vars_dict}
        # Combine the dictionaries
        vars_dict = {**cat_vars_dict, **unique_vars_dict}
        # Convert the dictionary to a DataFrame
        vars_df = pd.DataFrame(list(vars_dict.items()), columns=['Variable', 'Type'])
        self.variables_df = vars_df
            
    def create_emm_df(self, plus_2_stdev=False, minus_2_stdev=False):
        # Create a new DataFrame to store the EMM data
        emm_df = pd.DataFrame()

        # Iterate over each variable to set up the DataFrame
        for index, row in self.variables_df.iterrows():
            variable = row['Variable']
            var_type = row['Type']

            if var_type == 'categorical':
                # Set this column to unique values of the variable
                emm_df[variable] = self.data_df[variable].unique()
            elif var_type == 'continuous':
                # Create a list of unique values for each categorical variable
                categorical_vars = [col for col in self.variables_df['Variable'] if self.variables_df.loc[self.variables_df['Variable'] == col, 'Type'].values[0] == 'categorical']
                unique_combinations = self.data_df[categorical_vars].drop_duplicates()

                # Initialize an empty list to store the mean values
                mean_values = []

                # Iterate over unique combinations of categorical variable values
                for _, combination in unique_combinations.iterrows():
                    # Create a mask to select rows where categorical variable values match the combination
                    mask = True
                    for col in categorical_vars:
                        mask = mask & (self.data_df[col] == combination[col])

                    # Calculate the mean of the continuous variable within the selected rows
                    mean = self.data_df.loc[mask, variable].mean()
                    if plus_2_stdev:
                        mean = mean+self.data_df.loc[mask, variable].std()
                    if minus_2_stdev:
                        mean = mean-self.data_df.loc[mask, variable].std()
                    mean_values.append(mean)

                # Add the mean values to the corresponding rows in emm_df
                emm_df[variable] = mean_values
        self.emm_df = emm_df
    
    def define_design_matrix(self):
        """
        Defines the design matrix based on the patsy formula and returns it as a DataFrame.
        
        Parameters:
        - formula: str, the patsy formula to construct the design matrix
        - data_df: DataFrame, the data frame from which to construct the design matrix
        
        Returns:
        - The design matrix for the specified formula.
        """
        # Modify the formula to use a + b without specifying any dependent variable
        modified_formula = self.formula.split('~')[1].strip()
        
        # Create the design matrix
        self.emm_design_df = patsy.dmatrix(modified_formula, self.emm_df, return_type='dataframe')
    
    def predict_emm_design_df(self):
        self.emm_df['predictions'] = self.model.predict(self.emm_design_df)
        return self.emm_df
    
    def plot_predictions(self, out_dir=None):
        """
        Plots a grouped barplot using vars_df and emm_df to visualize the interaction between categorical variables.

        Parameters:
        - out_dir: str, optional, the directory to save the plot (default is None).
        """
        # Set the style for the plot
        emm_df = self.emm_df
        vars_df = self.variables_df
        sns.set(style="whitegrid")

        # Extract the categorical variable names from vars_df
        categorical_vars = vars_df[vars_df['Type'] == 'categorical']['Variable'].tolist()

        # Identify the continuous variable
        continuous_var = vars_df[vars_df['Type'] == 'continuous']['Variable'].iloc[0]

        # Create the barplot
        plt.figure(figsize=(10, 6))

        if len(categorical_vars) == 1:
            # If there's only one categorical variable, use it as x-axis without hue
            sns.barplot(x=categorical_vars[0], y='predictions', data=emm_df)
        elif len(categorical_vars) >= 2:
            # If there are two or more categorical variables, use 'hue' for the first one and 'col' for the rest
            sns.barplot(x=categorical_vars[0], y='predictions', hue=categorical_vars[1], col=categorical_vars[2:],
                        data=emm_df, col_wrap=3)  # Adjust col_wrap based on the number of variables

        # Add labels and title
        plt.xlabel(categorical_vars[0])
        plt.ylabel(continuous_var)
        plt.title(f'Grouped Barplot of EMM by Categorical Variables')

        # Show the legend
        if len(categorical_vars) >= 2:
            plt.legend(title=categorical_vars[1], loc='upper right')

        # Save the figure if out_dir is provided
        if out_dir:
            plt.savefig(f"{out_dir}/profile_plot.png", bbox_inches='tight')
            plt.savefig(f"{out_dir}/profile_plot.svg", bbox_inches='tight')
            print(f'Saved to {out_dir}/profile_plot.svg')

        # Show the plot
        plt.show()

class FactorialPlot(EMMPlot):
    
    def create_emm_df(self, plus_2_stdev=False, minus_2_stdev=False):
        # Create a new DataFrame to store the EMM data
        emm_df = pd.DataFrame()

        # Identify categorical variables
        categorical_vars = [var for var in self.variables_df['Variable'] if self.variables_df.loc[self.variables_df['Variable'] == var, 'Type'].values[0] == 'categorical']

        # Check if there are continuous variables
        continuous_vars = [var for var in self.variables_df['Variable'] if self.variables_df.loc[self.variables_df['Variable'] == var, 'Type'].values[0] == 'continuous']

        # If there are only categorical variables, generate all possible combinations
        if not continuous_vars:
            # Create a Cartesian product of all categorical variables
            all_combinations = pd.MultiIndex.from_product([self.data_df[cat_var].unique() for cat_var in categorical_vars], names=categorical_vars).to_frame(index=False)
            emm_df = all_combinations
        else:
            # Iterate over each variable to set up the DataFrame for continuous variables as before
            for continuous_var in continuous_vars:
                # Calculate the mean of the continuous variable within the selected rows, adjusted by standard deviation if specified
                unique_combinations = self.data_df[categorical_vars].drop_duplicates()
                for _, combination in unique_combinations.iterrows():
                    mask = True
                    for cat_var in categorical_vars:
                        mask = mask & (self.data_df[cat_var] == combination[cat_var])
                    mean = self.data_df.loc[mask, continuous_var].mean()
                    if plus_2_stdev:
                        mean += self.data_df.loc[mask, continuous_var].std() * 2
                    if minus_2_stdev:
                        mean -= self.data_df.loc[mask, continuous_var].std() * 2
                    emm_df = emm_df.append(combination.to_dict(), ignore_index=True)
                    emm_df.loc[emm_df.index[-1], continuous_var] = mean

        self.emm_df = emm_df

    def create_interaction_plot(self, factor_names, ms=10, ax=None):
        # Drop rows with NaN in any of the factor columns or in the predictions
        emm_df = self.emm_df
        emm_df = emm_df.dropna()
        display(emm_df)

        # Initialize the plotting area if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Determine the number of unique values for each factor
        unique_values_per_factor = [emm_df[fname].nunique() for fname in factor_names]

        # Map the first categorical variable to integers for the x-axis
        print(factor_names)
        x_factor = factor_names[0]
        x_levels = emm_df[x_factor].dropna().unique()
        x_levels.sort()  # Sort the levels to ensure consistent order
        x_mapping = {level: i for i, level in enumerate(x_levels)}
        emm_df['x_mapped'] = emm_df[x_factor].map(x_mapping)

        # Set colors and markers based on unique values
        colors = plt.cm.get_cmap('tab10', unique_values_per_factor[1])  # For the trace factor
        markers = ['o', 's', 'D', '^', 'v', '*', 'p', 'x']  # Default set of markers
        
        if len(factor_names) > 2 and unique_values_per_factor[2] > len(markers):
            markers *= (unique_values_per_factor[2] // len(markers)) + 1

        # Adjust the markers based on the third factor if it's provided
        marker_assignments = markers
        if len(factor_names) > 2:
            third_factor = factor_names[2]
            marker_assignments = emm_df[third_factor].map(dict(zip(emm_df[third_factor].unique(), markers)))

        # Generate the interaction plot
        fig = interaction_plot(
            x=emm_df['x_mapped'],
            trace=emm_df[factor_names[1]],
            response=emm_df['predictions'],
            colors=[colors(i) for i in range(unique_values_per_factor[1])],
            markers=marker_assignments,
            ms=ms,
            ax=ax
        )
        
        # Customize the plot
        ax.set_xticks(range(len(x_levels)))
        ax.set_xticklabels(x_levels)
        ax.set_title('Interaction Plot of Predicted EMMs')
        ax.set_xlabel(x_factor)
        ax.set_ylabel('Predictions')
        ax.legend(title=factor_names[1])
        
        # Display the plot
        plt.show()

import matplotlib.pyplot as plt
import pandas as pd

class EstimatedMarginalMean(FactorialPlot):

    # def create_interaction_plot_dynamic(self, emm_df, categorical_vars, ms=10, out_dir=None):
    #     """
    #     Create an interaction plot for 2 or 3 categorical variables in emm_df.

    #     Parameters:
    #     - emm_df: DataFrame containing the data
    #     - categorical_vars: List of strings specifying the categorical variables
    #     - ms: Marker size for the plot (default is 10)
    #     """
    #     # Check if we have 2 or 3 categorical variables
    #     if len(categorical_vars) == 2:
    #         self.plot_interaction_2_vars(emm_df, categorical_vars, ms, out_dir)
    #     elif len(categorical_vars) == 3:
    #         self.plot_interaction_3_vars(emm_df, categorical_vars, ms, out_dir)
    #     else:
    #         print("This function supports only 2 or 3 categorical variables.")

    def plot_interaction_2_vars(self, emm_df, categorical_vars, ms, out_dir=None):
        # Customize your plot here for 2 categorical variables
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create an interaction plot using Matplotlib
        city_colors = plt.cm.get_cmap('tab10', len(emm_df[categorical_vars[1]].unique()))
        city_color_mapping = {city: color for city, color in zip(emm_df[categorical_vars[1]].unique(), city_colors(range(len(emm_df[categorical_vars[1]].unique()))))}

        for city in emm_df[categorical_vars[1]].unique():
            for age_group in emm_df[categorical_vars[2]].unique():
                subset = emm_df[(emm_df[categorical_vars[1]] == city) & (emm_df[categorical_vars[2]] == age_group)]

                label = f'{categorical_vars[1]}: {city}, {categorical_vars[2]}: {age_group}'
                color = city_color_mapping[city]

                ax.plot(subset[categorical_vars[0]], subset['predictions'], marker='o', label=label, color=color, ms=ms)

        # Create the first legend with a single item for each city
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = list(set(labels))
        unique_handles = [handles[labels.index(label)] for label in unique_labels]
        ax.legend(unique_handles, unique_labels, title=categorical_vars[1], loc='upper left')

        # Customize the plot
        ax.set_xlabel(categorical_vars[0])
        ax.set_ylabel('Predictions')
        ax.set_title(f'Interaction Plot of Predicted EMMs ({categorical_vars[1]})')

        # Display the plot
        plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
        plt.tight_layout()
        plt.savefig(f"{out_dir}/interaction_plot.png", bbox_inches='tight')
        plt.savefig(f"{out_dir}/interaction_plot.svg", bbox_inches='tight')
        print(f'Saved to {out_dir}/interaction_plot.svg')
        plt.show()

    def plot_interaction_3_vars(self, emm_df, categorical_vars, ms, out_dir=None):
        # Customize your plot here for 3 categorical variables
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create an interaction plot using Matplotlib with line hatching
        city_colors = plt.cm.get_cmap('tab10', len(emm_df[categorical_vars[1]].unique()))
        city_color_mapping = {city: color for city, color in zip(emm_df[categorical_vars[1]].unique(), city_colors(range(len(emm_df[categorical_vars[1]].unique()))))}

        line_styles = ['-', '--']
        markers = ['o', 's', 'D', '^', 'v', '*', 'p', 'x']

        for city in emm_df[categorical_vars[1]].unique():
            for age_group in emm_df[categorical_vars[2]].unique():
                subset = emm_df[(emm_df[categorical_vars[1]] == city) & (emm_df[categorical_vars[2]] == age_group)]

                label = f'{categorical_vars[1]}: {city}, {categorical_vars[2]}: {age_group}'
                color = city_color_mapping[city]
                linestyle = line_styles[emm_df[categorical_vars[2]].unique().tolist().index(age_group)]
                marker = markers[emm_df[categorical_vars[2]].unique().tolist().index(age_group)]

                ax.plot(subset[categorical_vars[0]], subset['predictions'], marker=markers[0], label=label, linestyle=linestyle, color=color, ms=ms)

        # Create the first legend with a single item for each city
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = list(set(labels))
        unique_handles = [handles[labels.index(label)] for label in unique_labels]
        ax.legend(unique_handles, unique_labels, title=categorical_vars[1], loc='upper left')

        # Create the second legend for Age_Group
        age_legend_labels = [f'{categorical_vars[2]}: {age_group}' for age_group in emm_df[categorical_vars[2]].unique()]
        age_legend_handles = [plt.Line2D([0], [0], linestyle=line_styles[i], color='black') for i in range(len(age_legend_labels))]
        ax2 = plt.gca().add_artist(ax.legend(age_legend_handles, age_legend_labels, title=categorical_vars[2], loc='upper right'))

        # Third legend: City colors
        city_legend_labels = [f'{city}' for city in emm_df[categorical_vars[1]].unique()]
        city_legend_handles = [plt.Line2D([0], [0], color=city_color_mapping[city], linestyle='-', marker='o') for city in emm_df[categorical_vars[1]].unique()]
        ax.legend(city_legend_handles, city_legend_labels, title='City Colors', loc='lower left')


        # Customize the plot
        ax.set_xlabel(categorical_vars[0])
        ax.set_ylabel('Predictions')
        ax.set_title(f'Interaction Plot of Predicted EMMs ({categorical_vars[1]})')

        # Display the plot
        plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
        plt.tight_layout()
        plt.savefig(f"{out_dir}/interaction_plot.png", bbox_inches='tight')
        plt.savefig(f"{out_dir}/interaction_plot.svg", bbox_inches='tight')
        print(f'Saved to {out_dir}/interaction_plot.svg')
        plt.show()

class PartialRegressionPlot():
    def __init__(self, model, out_dir=None, design_matrix=None, palette='Greys'):
        self.out_dir = out_dir
        self.model = model
        self.design_matrix = design_matrix
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
        sns.set_palette(palette, 1, desat=1)
        sns.set_style('white')

    def plot_and_save_partregress(self):
        # Adjust the size of the figure based on the number of columns
        if self.design_matrix is not None:
            num_cols = len(self.design_matrix.columns)
            fig_size = ((num_cols*2), (num_cols*2))  
            plt.figure(figsize=fig_size)
            sm.graphics.plot_partregress_grid(self.model, fig=plt.gcf())
        else:
            sm.graphics.plot_partregress_grid(self.model)
        
        if self.out_dir:
            # Save the figure in PNG and SVG formats
            plt.savefig(f"{self.out_dir}/partial_regression_plot.png", bbox_inches='tight')
            plt.savefig(f"{self.out_dir}/partial_regression_plot.svg", bbox_inches='tight')
            print(f'Saved to {self.out_dir}/partial_regression_plot.svg')

        # Display the plot
        plt.show()
        return plt
    
    def run(self):
        plt = self.plot_and_save_partregress()
        return plt

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

    def figure_saver(self, name=''):
        '''
        Method to save the forest plot.
        '''
        # Save the plot as PNG and SVG
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            self.fig.savefig(os.path.join(self.out_dir, f"{name}_regression_forest_plot.png"), bbox_inches='tight')
            self.fig.savefig(os.path.join(self.out_dir, f"{name}_regression_forest_plot.svg"), bbox_inches='tight')
            print(f'Saved to {self.out_dir} as regression_forest_plot.svg and .png')

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

    def create_and_display_forest_plot(self):
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
                pval="pvalue",  # Column of p-value to be reported on right
                # groupvar="group",
                
                #Specialized Annotation
                # annote=["estimate"],  # columns to report on left of plot
                # annoteheaders=["Est.(95% Conf. Int.)"],
                # rightannote=["pvalue"],  # columns to report on right of plot 
                # right_annoteheaders=["P-value"],

                # Axis Labels
                xlabel="Regression coefficient",
                ylabel='Est.(95% Conf. Int.)',
                
                #   # Forest Plot Configuration
                decimal_precision=self.sig_digits,
                capitalize='capitalize',
                color_alt_rows=False,  # Gray alternate rows
                table=self.table,
                flush=False,
                
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

    def run(self):
        '''
        Orchestrator method.
        '''
        self.prepare_results()
        self.table_prep()
        self.create_and_display_forest_plot()
        self.figure_saver()

class MultinomialForestPlot(ForestPlot):
    '''
    This class takes a fitted multinomial statsmodels model object (MNLogit)
    and generates a forest plot for the regression coefficients of each class
    with respect to the reference class.
    '''
    def prepare_multinomial_results_and_plot(self):
        # This will create a forest plot for each class in the multinomial model
        for i in range(1, self.model.J):
            # Conf intervals for this class
            class_labels = self.model.conf_int().index.get_level_values(0).unique()
            conf = self.model.conf_int().loc[class_labels[i-1]]
            
            # Coefficients, confidence intervals, and p-values for the i-th class vs reference
            params = self.model.params.iloc[:, i - 1]

            # Check if pvalues is a DataFrame and extract appropriately
            pvalues = self.model.pvalues.iloc[:, i - 1] if isinstance(self.model.pvalues, pd.DataFrame) else self.model.pvalues

            self.data_for_plot = pd.DataFrame({
                'estimate': params,
                'll': conf.iloc[:, 0],
                'hl': conf.iloc[:, 1],
                'label': params.index,
                'pvalue': pvalues
            })
            self.create_and_display_forest_plot()
            print("----Forest Plot For: " + class_labels[i-1] + " ----")
            self.figure_saver(name=class_labels[i-1])
            
    def run(self):
        self.prepare_multinomial_results_and_plot()
   
class PartialDependencePlot(EMMPlot):
    """
    This is a partial dependence plot class. 
    It will hold all variables except one at their means, and vary the last across its range in incremental steps. 
    It will then make predictions, seeing how the prediction changes as a a funciton of this one variable.
    
    This class is generated for use in logistic regressions. 
    It will create a dictionary containing one dataframe per each classification.
    Each dataframe will contain the predictions for each classification and the dynamic range of data. 
    """
    def __init__(self, formula, data_df, model, design_matrix, outcomes_df=None, data_range=None, out_dir=None, marginal_method='mean', debug=False):
        """
        Args:
            formula: str
                - this is the formula provided to the regression method 
            data_df: pd.DataFrame
                - this is the dataframe which was contains all your regressors and your observations.
            model: StatsModels.Model instance
                - this is the model which was fitted to the regression method
            design_matrix: pd.DataFrame
                - This is the dataframe acting as the design matrix generated in my regression notebooks.
            outcomes_df: pd.DataFrame (Optional)
                - this is the dataframe which contains the observations. 
            data_range: tuple (optional)
                - This specifies the maximum and minimum values to vary the variable of interest over. 
                    If not entered, will default to minimum and maximum values of the variable. 
            out_dir: string (optional)
                - If set, this will be the directory where the figures are saved. 
            marginal_method = str | mean | min | max | absmin | absmax
                - This determines how we will hold all variables aside from the variable of interest.
                    We can set them to their mean, their minimum, or their maximum. 
                    absmin and absmax will set the value to the absolute lowest value in the dynamic range. 
                    min and max will set the value to the min or max of the observed input value for that variable.
        """
        super().__init__(formula, data_df, model)
        self.design_matrix = design_matrix
        self.outcomes_df = outcomes_df
        self.results_dict = dict()
        self.data_range = data_range
        self.out_dir = out_dir
        self.marginal_method = marginal_method
        self.debug = debug
        
    def partial_dependence_plots(self):
        """
        This method will create a figure for each classification. Each figure will include lines for each variable,
        showing how the prediction changes with the variable, allowing for comparison of variable effects within each classification.
        """
        num_variables = len(self.variables_df['Variable'])
    
        # Selecting the color palette by number of variables
        if num_variables <= 10:
            palette = sns.color_palette("tab10", n_colors=num_variables)
            sns.set_palette(palette, desat=1)
        elif num_variables <= 20:
            palette = sns.color_palette("tab20", n_colors=num_variables)
            sns.set_palette(palette, desat=1)
        else:
            # Creating a colormap for more than 20 variables
            cmap = plt.cm.get_cmap('hsv', num_variables)
            palette = [cmap(i) for i in range(num_variables)]
            sns.set_palette(palette, desat=1)
            
        # Generate the Plots
        for classification, results_df in self.results_dict.items():
            plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
            plt.title(f"Partial Dependence Plot for {classification}")
            
            for variable in self.variables_df['Variable']:
                sns.lineplot(x=results_df[f'{variable}'], y=results_df[f'{variable}_predictions'], label=variable, palette=palette)
            
            plt.xlabel("Variable Value")
            plt.ylabel("Predicted Probability")
            plt.legend(title="Variable")
            if self.out_dir is not None:
                plt.savefig(os.path.join(self.out_dir, f'pdp_{classification}.svg'))
                plt.savefig(os.path.join(self.out_dir, f'pdp_{classification}.png'))
            plt.show()
            
    def partial_dependence_prediction(self, partial_dependence_df, variable_name, prediction_index):
        """
        This method will take the partial_dependence_df and make a prediction on it.
        The predictions will be stored in self.results_df, as will the range of the continuous variable. 
        
        The hard part, especially for a multinomial logit, is that each prediction yields N predictions. 
        The maximal prediction is the model's 'choice', but this is not correct for partial dependence. 
        We need to find the value of the prediction at the index of our predictor. 
        """
        
        # This will return a pandas dataframe of predictions
        predictions = self.model.predict(partial_dependence_df)
        # This gets the prediction associated with our class of interest
        self.results_df[f'{variable_name}_predictions'] = predictions.loc[:, prediction_index]
        self.results_df[variable_name] = partial_dependence_df.loc[:, variable_name]

    def partial_dependence_df(self, variable_name:str, variable_type:str):
        """
        This method will take the variable, the type, and create a new dataframe based upon it. 
        This dataframe will be used for predictions.
        
        Essentially, the variable of interest varies from min to max in 100 steps. 
        All of the other variables are held at their means. 
        For categorical features, the partial dependence is very easy to calculate. 
        
        TODO:
        For each of the categories, we get a PDP estimate by forcing all data instances to have the same category.
        For example, if we look at the bike rental dataset and are interested in the partial dependence plot for the season, 
        we get four numbers, one for each season. To compute the value for “summer”, we replace the season of all data instances
        with “summer” and average the predictions.
        """
        if variable_type != 'continuous':
            raise ValueError(f"{variable_type} variables are not supported yet. Please use continuous variables instead.")
        
        # Intialize the dataframe
        partial_dependence_df = pd.DataFrame(np.nan, index=range(100), columns=self.variables_df['Variable'])
        partial_dependence_df['Intercept'] = 1
        for variable in self.variables_df['Variable']:
            # Range our variable of interest across a dynamic range. 
            if variable==variable_name:
                if self.data_range is None: 
                    partial_dependence_df[variable_name] = np.linspace(np.min(self.data_df[variable_name]), np.max(self.data_df[variable_name]), 100)
                else:
                    partial_dependence_df[variable_name] = np.linspace(self.data_range[0], self.data_range[1], 100)
            # Hold variables which are not our variables of interest constant at mean
            else:
                if self.marginal_method == 'mean':
                    partial_dependence_df[variable] = np.mean(self.data_df[variable])  
                elif self.marginal_method == 'min':
                    partial_dependence_df[variable] = np.min(self.data_df[variable])  
                elif self.marginal_method == 'max':
                    partial_dependence_df[variable] = np.max(self.data_df[variable])  
                elif self.marginal_method == 'absmin':
                    partial_dependence_df[variable] = self.data_range[0] if self.data_range is not None else np.min(self.data_df)  
                elif self.marginal_method == 'absmax':
                    partial_dependence_df[variable] = self.data_range[1] if self.data_range is not None else np.max(self.data_df)  
                else:
                    raise TypeError("Invalid marginal method: %s" % self.marginal_method)       
        partial_dependence_df = partial_dependence_df.loc[:, self.design_matrix.columns]
        return partial_dependence_df   
    
    def orchestrate_marginal_predictions(self):
        """
        This method is going to iterate over each variable, getting its estimated marginal means. 
        We will pass each variable, identifying it as the variable to vary. 
        We will flag the type of the variable, continuous or categorical. 
        
        Notes:
        We need to generate a dataframe of results for each potential class of observations. 
        There are only >1  classes of observations in  binomial logits.
        There are only >2 classes of observations in multinomial logits and MANOVAs or GLMs with numerous prediction classes.
        """
        for i, observation in enumerate(self.outcomes_df.columns):
            print(f"{i}:{observation}")
            # Initialize a new DF for each classification
            self.results_df = pd.DataFrame()
            for index, row in self.variables_df.iterrows():
                # Set up the Data
                partial_dependence_df = self.partial_dependence_df(row['Variable'], row['Type'])
                # Predict the data and assign it to the results_df
                self.partial_dependence_prediction(partial_dependence_df, row['Variable'], i)
            # Store results DF for this classification
            self.results_dict[observation] = self.results_df
            
    def validate_model(self):
        """
        This is an initial check to make sure we are okay to begin. 
        If the appropriate data for the model has been entered, we will then relate the index of predictions
        to the class of the prediction. 
        """
        if isinstance(self.model, smd.MNLogit) or isinstance(self.model, smd.Logit):
            if self.outcomes_df is None:
                raise TypeError("Logistic model detected, but outcomes_df is none. \n You must enter the dataframe containing the observations. ")
                    
    def run(self):
        """
        Orchestration Method
        """
        self.validate_model()
        self.extract_unique_variables()
        self.orchestrate_marginal_predictions()
        self.partial_dependence_plots()

class GLMMarginalsPlot():
    """
    The GLMMarginalsPlot class generates profile plots to visualize the effect of varying one or more predictor variables 
    on the response variable, holding other variables constant. This class is particularly useful for exploring and 
    presenting the relationship between predictors and the predicted outcome in a statistical model.
    
    This best takes in a GLM, or data which has at least one continuous predictor variable. 
    
    AKA a profile plot.

    Attributes:
        formula (str): The regression formula used in the model.
        data_df (pd.DataFrame): The DataFrame containing the regressors and observations.
        model (StatsModels.Model instance): The fitted statistical model.
        marginal_scenarios_dict (dict): A dictionary specifying the categories (predictors) and their values to be 
            used for generating marginal scenarios. For continuous variables, the value should be 'continuous', 
            and for categorical variables, it should be a list of values to iterate over.
            Example: {'age': 'continuous', 'gender': ['male', 'female'], 'dose': 'std'}
            For 'continuous'
                - will predict across the entire range of values, spanning data_range
            For 'mean'
                - will find the mean value of the predictor within this combination of other levels and derive estimates at it
            For 'std'
                - will find the mean value of the predictor within this combination of other levels and derive estimates of +/-2stdevs
        data_range (tuple, optional): Specifies the range (min, max) to vary continuous variables. If not provided, 
            the range is derived from the data.
            - This is only used when 'continuous' is specified.
        out_dir (str, optional): Directory path to save the generated plots.
        variance_bars (str) sem | 95ci: how to plot the error bars. Set to None for no error bars.
        debug (bool): If True, enables debug mode for additional logging.

    Methods:
        set_palette: Sets the color palette for the plot based on the number of scenarios.
        profile_plot: Generates a single profile plot with traces for each marginal prediction.
        marginal_prediction: Makes predictions for a given design matrix and scenario, storing the results.
        define_design_matrix: Constructs the design matrix from a given DataFrame based on the regression formula.
        generate_marginal_scenario_dataframe: Modifies the data DataFrame according to the current scenario.
        generate_scenarios: Generates all possible scenarios from `marginal_scenarios_dict`.
        orchestrate_marginal_predictions: Orchestrates the generation of scenarios, prediction, and plotting.
        run: Executes the orchestration method to generate and plot marginal predictions.

    Notes:
        - The `marginal_scenarios_dict` should be carefully set up to include all predictors you wish to analyze. 
          Continuous variables should have the value 'continuous', and categorical variables should list all categories 
          you wish to iterate over.
        - Profile plots are useful for understanding the behavior of the model across different levels of one or more 
          predictors. Each trace on the plot represents the predicted outcome as one predictor varies, holding others constant.
        - Ensure that the `model` is compatible with the design matrices generated by this class.
    
    TODO:
        -   Store the confidence intervals from the predictions. Plot them. 
    """
    def __init__(self, formula, data_df, model, marginal_scenarios_dict, data_range=None, out_dir=None, variance_bars='sem', debug=False):
        """
        init method
        """
        # super().__init__(formula, data_df, model)
        self.formula = formula
        self.data_df=data_df
        self.model=model
        self.results_dict = dict()
        self.data_range = data_range
        self.out_dir = out_dir
        self.marginal_scenarios_dict = marginal_scenarios_dict
        self.debug = debug
        self.variance_bars = variance_bars
        self.results_df = pd.DataFrame()
        
    def set_palette(self):
        num_scenarios = len(self.marginal_scenarios)
        if num_scenarios <= 10:
            palette = sns.color_palette('tab10', len(self.marginal_scenarios)) 
            sns.set_palette('tab10')
        elif num_scenarios <= 20:
            palette = sns.color_palette('tab10', len(self.marginal_scenarios)) 
            sns.set_palette('tab20')
        else:
            cmap = plt.cm.get_cmap('hsv', num_scenarios)
            palette = [cmap(i) for i in range(num_scenarios)]
            sns.set_palette(palette)
        return palette

    def get_continuous_bars(self, scenario_label):
        """
        Method to extract error estiamtes or pass on them. 
        """
        # Prepares error bars
        if self.variance_bars=='sem':
            lower = self.results_df[f'{scenario_label}_predictions'] - self.results_df[f'{scenario_label}_sem']
            higher = self.results_df[f'{scenario_label}_predictions'] + self.results_df[f'{scenario_label}_sem']
        # Confidence interval as error bars
        elif self.variance_bars=='95ci':
            lower =  self.results_df[f'{scenario_label}_ci_low']
            higher = self.results_df[f'{scenario_label}_ci_high']
        # No error bars
        else:
            lower = None; higher = None
        return lower, higher
    
    def get_std_bars(self, scenario_label):
        if self.variance_bars=='sem':
            bars = self.results_df[f'{scenario_label}_sem']
        elif self.variance_bars=='95ci':
            higher = self.results_df[f'{scenario_label}_predictions'] + self.results_df[f'{scenario_label}_sem']
            mean_predictions = self.results_df[f'{scenario_label}_predictions']
            bars = higher - mean_predictions
        else:
            bars = None
        return bars
    
    def profile_plot(self):
        """
        This method will create a profile plot.
        each marginal plot will have a trace for each marginal prediction. 
        """
        palette = self.set_palette()
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        plt.title(f"Marginals Plot")
        
        for i, marginal_scenario in enumerate(self.marginal_scenarios):
            scenario_label = "_".join([f"{k}={v}" for k, v in marginal_scenario.items()])
            if 'continuous' in marginal_scenario.values():
                sns.lineplot(x=self.results_df[f'{scenario_label}_observations'], y=self.results_df[f'{scenario_label}_predictions'], label=scenario_label)
                lower, higher = self.get_continuous_bars(scenario_label)
                if lower is not None:
                    plt.fill_between(self.results_df[f'{scenario_label}_observations'], lower, higher, color=palette[i], alpha=0.2)
            elif 'std' in marginal_scenario.values() or 'mean' in marginal_scenario.values():
                x_values = self.results_df[f'{scenario_label}_observations']
                y_values = self.results_df[f'{scenario_label}_predictions']
                error_bars = self.get_std_bars(scenario_label)
                if self.variance_bars is not None:
                    plt.errorbar(x=x_values, y=y_values, yerr=error_bars, fmt='-o', label=scenario_label, color=palette[i], capsize=5)
                else:
                    plt.scatter(x=x_values, y=y_values, fmt='o', label=scenario_label, color=palette[i])
            else:
                raise ValueError("Unexpected value entered. At least one key must be std | mean | continuous")
        
        plt.xlabel("Observed Value")
        plt.ylabel("Predicted Value")
        plt.legend(title="Variable")
        if self.out_dir is not None:
            plt.savefig(os.path.join(self.out_dir, f'glm_marginal_plot.svg'))
            plt.savefig(os.path.join(self.out_dir, f'glm_marginal_plot.png'))
        plt.show()

    def marginal_prediction(self, marginal_design_matrix, marginal_scenario, continuous_values):
        """
        This method will take the marginal_design_matrix and make a prediction on it. 
        It will store its results in a dataframe labelled by the marginal_scenario. 
        """
        try:
            predictions = self.model.get_prediction(marginal_design_matrix)
            scenario_label = "_".join([f"{k}={v}" for k, v in marginal_scenario.items()])
            self.results_df[f'{scenario_label}_predictions'] = predictions.predicted_mean
            self.results_df[f'{scenario_label}_observations'] = continuous_values[range(len(predictions.predicted_mean))]
            
            self.results_df[f'{scenario_label}_sem'] =  predictions.se_mean
            
            ci = predictions.conf_int()
            self.results_df[f'{scenario_label}_ci_low'] = ci[:, 0]
            self.results_df[f'{scenario_label}_ci_high'] = ci[:, 1]
        except Exception as e:
            print("*** \n Error occurred while using get_prediction method: " + str(e) + '\n Defalting to .predict() method. Losing estimate of error.')
            predictions = self.model.predict(marginal_design_matrix)
            scenario_label = "_".join([f"{k}={v}" for k, v in marginal_scenario.items()])
            self.results_df[f'{scenario_label}_predictions'] = predictions
            self.results_df[f'{scenario_label}_observations'] = continuous_values[0:len(predictions)]

    def define_design_matrix(self, marginal_df):
        """
        Defines the design matrix based on the patsy formula and returns it as a DataFrame.
        
        Parameters:
        - marginal_df: DataFrame, the data frame from which to construct the design matrix
        
        Returns:
        - Tuple containing the design matrix for the dependent variable and the design matrix for the independent variables.
        """
        y, X = patsy.dmatrices(self.formula, marginal_df, return_type='dataframe')
        return y, X   
    
    def extract_mean_value(self, scenario, target_variable):
        # Filter data based on the scenario criteria, excluding the target variable 'mean' situation
        filtered_df = self.data_df.copy()
        for key, value in scenario.items():
            if key != target_variable and key in filtered_df.columns:
                if isinstance(value, list):
                    filtered_df = filtered_df[filtered_df[key].isin(value)]
                else:
                    filtered_df = filtered_df[filtered_df[key] == value]
        
        # Calculate the mean for the target variable in the filtered DataFrame
        mean = filtered_df[target_variable].mean()
        std = filtered_df[target_variable].std()
        return mean, std
    
    def generate_marginal_scenario_dataframe(self, current_marginal_scenario):
        """
        Modify self.data_df based on the current scenario in current_marginal_scenario. For continuous keys,
        set the column to a range between self.data_range[0] and self.data_range[1] in 100 steps.
        """
        # Make a copy of the original DataFrame
        marginal_df = self.data_df.copy()
        # Iterate over each key-value pair in the current scenario
        continuous_values = None
        for key, value in current_marginal_scenario.items():
            # Make marginal predictions over a range of data
            if value == 'continuous':
                marginal_df = marginal_df.iloc[0:100,:]
                if self.data_range is not None:
                    continuous_values = np.linspace(self.data_range[0], self.data_range[1], len(marginal_df))
                else:
                    continuous_values = np.linspace(self.data_df[key].min(), self.data_df[key].max(), len(marginal_df))
                marginal_df[key] = continuous_values
            # Make marginal predictions at standard deviations
            elif value == 'std':
                marginal_df = marginal_df.iloc[0:2,:]
                mean, std = self.extract_mean_value(current_marginal_scenario, key)  
                continuous_values = np.array([mean - 2*std, mean + 2*std])
                marginal_df[key] = continuous_values
            # Generate Estimated Marginal Means
            elif value == 'mean':
                marginal_df = marginal_df.iloc[0:1,:]
                mean, _ = self.extract_mean_value(current_marginal_scenario, key)  
                continuous_values = np.array([mean]) 
                marginal_df[key] = continuous_values        
            else: 
                marginal_df[key] = value
        return marginal_df, continuous_values
    
    def generate_scenarios(self):
        """
        For each category, set it to one of its possible values, creating a scenario. 
        Each scenario is a dictionary where keys are categories and values are the set values for those categories.
        Pass each scenario as a dictionary to self.generate_design_matrix.
        """
        # Initialize a list to hold the scenarios
        scenarios = [{}]

        # Iterate over each category and its values in the dictionary
        for category, values in self.marginal_scenarios_dict.items():
            # Prepare the next set of scenarios
            next_scenarios = []
            # Handle continuous variables by setting a placeholder or a specific value
            if values == 'continuous':
                pass

            # Expand the scenarios with this category's values
            for scenario in scenarios:
                for value in values:
                    # Create a new scenario based on the existing one, with the current category set to the current value
                    updated_scenario = scenario.copy()
                    updated_scenario[category] = value
                    next_scenarios.append(updated_scenario)

            # Update the scenarios list with the expanded set of scenarios
            scenarios = next_scenarios
        self.marginal_scenarios = scenarios
        
    def orchestrate_marginal_predictions(self):
        """
        This method will take the dictionary specifying the marginal values and iterate over it.
        It will have a design martrix created for each marginal dataframe. 
        It will predict and store the marginal dataframe. 
        """
        self.generate_scenarios()
        for marginal_scenario in self.marginal_scenarios:
            marginal_df, continuous_values = self.generate_marginal_scenario_dataframe(marginal_scenario)
            _, marginal_design_matrix = self.define_design_matrix(marginal_df)
            self.marginal_prediction(marginal_design_matrix, marginal_scenario, continuous_values)
        
    def run(self):
        self.orchestrate_marginal_predictions()
        self.profile_plot()

##
def predict_outcomes(model, design_matrix):
    """
    Predicts the outcomes for each row in the data DataFrame based on the given model and formula.

    Parameters:
    - model: The fitted OLS model.
    - data_df: DataFrame containing the data for prediction.

    Returns:
    - predictions: A DataFrame with predicted values for each row in the data_df.
    """
    # Add a column for predictions to the data_df
    
    return model.predict(design_matrix)

def calculate_average_predictions(data_df, predictions, group1_column, group2_column):
    """
    Calculates average predictions for each combination of unique values in Group 1 and Group 2.

    Parameters:
    - data_df: DataFrame containing the 'Group 1' and 'Group 2' columns.
    - predictions: Series containing the predicted values.
    - group1_column: Name of the Group 1 column in data_df.
    - group2_column: Name of the Group 2 column in data_df.

    Returns:
    - average_predictions_df: DataFrame with 'Group 1', 'Group 2', and 'Average Prediction' columns.
    """
    # Create an empty DataFrame to store the results
    average_predictions_df = pd.DataFrame(columns=['Group 1', 'Group 2', 'Average Prediction'])

    # Get unique values in Group 1 and Group 2
    unique_group1_values = data_df[group1_column].unique()
    unique_group2_values = data_df[group2_column].unique()

    # Loop through unique values of Group 1 and Group 2
    for group1_value in unique_group1_values:
        for group2_value in unique_group2_values:
            # Filter predictions based on the indices where Group 1 and Group 2 match
            filtered_predictions = predictions[(data_df[group1_column] == group1_value) & (data_df[group2_column] == group2_value)]

            # Calculate the average prediction for this combination
            average_prediction = filtered_predictions.mean()

            # Add the result to the DataFrame
            average_predictions_df = average_predictions_df.append({'Group 1': group1_value, 'Group 2': group2_value, 'Average Prediction': average_prediction}, ignore_index=True)

    return average_predictions_df

def plot_grouped_barplot(data_df, out_dir):
    """
    Plots a grouped barplot with 'Group 2' on the x-axis and 'Group 1' bars side-by-side.

    Parameters:
    - data_df: DataFrame containing the data for plotting.
    """
    # Set the style for the plot
    sns.set(style="whitegrid")

    # Create the barplot
    plt.figure(figsize=(20, 6))
    sns.barplot(x='Group 2', y='Average Prediction', hue='Group 1', data=data_df)

    # Add labels and title
    plt.xlabel('Group 2')
    plt.ylabel('Average Prediction')
    plt.title('Grouped Barplot of Average Predictions')

    # Show the legend
    plt.legend(title='Group 1', loc='upper right')
    
    # Save the figure
    plt.savefig(f"{out_dir}/estimated_marginal_mean.png", bbox_inches='tight')
    plt.savefig(f"{out_dir}/estimated_marginal_mean.svg", bbox_inches='tight')
    print(f'Saved to {out_dir}/estimated_marginal_mean.svg')

    # Show the plot
    plt.show()
    
def calculate_average_actual(data_df, group1_column, group2_column, outcome_column):
    """
    Calculates average actual outcomes for each combination of unique values in Group 1 and Group 2.

    Parameters:
    - data_df: DataFrame containing the data.
    - group1_column: Name of the Group 1 column in data_df.
    - group2_column: Name of the Group 2 column in data_df.
    - outcome_column: Name of the column containing the actual outcomes.

    Returns:
    - average_actual_df: DataFrame with 'Group 1', 'Group 2', and 'Average Actual' columns.
    """
    # Create an empty DataFrame to store the results
    average_actual_df = pd.DataFrame(columns=['Group 1', 'Group 2', 'Average Actual'])

    # Get unique values in Group 1 and Group 2
    unique_group1_values = data_df[group1_column].unique()
    unique_group2_values = data_df[group2_column].unique()

    # Loop through unique values of Group 1 and Group 2
    for group1_value in unique_group1_values:
        for group2_value in unique_group2_values:
            # Filter data based on the indices where Group 1 and Group 2 match
            filtered_data = data_df[(data_df[group1_column] == group1_value) & (data_df[group2_column] == group2_value)]

            # Calculate the average actual outcome for this combination
            average_actual = filtered_data[outcome_column].mean()

            # Add the result to the DataFrame
            average_actual_df = average_actual_df.append({'Group 1': group1_value, 'Group 2': group2_value, 'Average Actual': average_actual}, ignore_index=True)

    return average_actual_df


def plot_grouped_barplot_actual(data_df, out_dir):
    """
    Plots a grouped barplot with 'Group 2' on the x-axis and 'Group 1' bars side-by-side for average actual outcomes.

    Parameters:
    - data_df: DataFrame containing the data for plotting.
    """
    # Set the style for the plot
    sns.set(style="whitegrid")

    # Create the barplot
    plt.figure(figsize=(20, 6))
    sns.barplot(x='Group 2', y='Average Actual', hue='Group 1', data=data_df)

    # Add labels and title
    plt.xlabel('Group 2')
    plt.ylabel('Average Actual Outcome')
    plt.title('Grouped Barplot of Average Actual Outcomes')

    # Show the legend
    plt.legend(title='Group 1', loc='upper right')

    # Show the plot
    plt.show()
    
    # Save the figure
    out_dir = out_dir # Replace with your output directory
    plt.savefig(f"{out_dir}/actual_marginal_mean.png", bbox_inches='tight')
    plt.savefig(f"{out_dir}/actual_marginal_mean.svg", bbox_inches='tight')
    print(f'Saved to {out_dir}/actual_marginal_mean.svg')
    
def run_ancova_emm(model, df, design_matrix, group_1, group_2, outcome_column, out_dir=None):
    """
    Executes the ANCOVA EMM analysis pipeline, generating predictions based on a given model and plotting 
    both the average predicted and actual outcomes for specified groupings within the data.

    This function serves as a wrapper to streamline the process of predicting outcomes using the model,
    calculating average predictions and actual outcomes for specified group categories, and then plotting
    these averages for visual comparison.
    
    This is a type 1 EMM, which is techncially more accurate. It makes predictions on each observation, then averages them together. 

    Parameters:
    - model: A fitted statistical model object that has a predict method. This model is used to generate 
             outcome predictions based on the design matrix provided.
    - design_matrix: DataFrame, the design matrix containing data for prediction. This includes both 
                     the group categorization columns and any other predictors required by the model.
    - group_1: String, the name of the first grouping variable in the design matrix. This categorizes data 
               into different groups for which EMMs are to be calculated.
    - group_2: String, the name of the second grouping variable in the design matrix, used in conjunction 
               with group_1 to further categorize data.
    - outcome_column: String, the name of the column in the design matrix that contains the actual outcome 
                      values. These are used for calculating and plotting actual means.
    - out_dir: String, optional, the directory path where the plots will be saved. If None, plots will 
               not be saved to files.

    Outputs:
    The function generates two plots: one for the average predictions and another for the average actual 
    outcomes, grouped by the specified categorization variables. Optionally, these plots are saved to the 
    specified output directory.

    Returns:
    None
    """
    predictions = predict_outcomes(model=model, data_df=design_matrix)
    average_predictions_df = calculate_average_predictions(df, predictions, group_1, group_2)
    print("Estimated Marginal means below")
    plot_grouped_barplot(average_predictions_df, out_dir=out_dir)
    print('Actual means below')
    plot_grouped_barplot_actual(calculate_average_actual(df, group_1, group_2, outcome_column),out_dir=out_dir)