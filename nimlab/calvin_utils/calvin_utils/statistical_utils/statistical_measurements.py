import os
import patsy 
import re
import numpy as np
from statsmodels.graphics.factorplots import interaction_plot


import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import maybe_unwrap_results
from typing import Type
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

class FlexibleInteractionPlot(FactorialPlot):

    def create_interaction_plot_dynamic(self, emm_df, categorical_vars, ms=10, out_dir=None):
        """
        Create an interaction plot for 2 or 3 categorical variables in emm_df.

        Parameters:
        - emm_df: DataFrame containing the data
        - categorical_vars: List of strings specifying the categorical variables
        - ms: Marker size for the plot (default is 10)
        """
        # Check if we have 2 or 3 categorical variables
        if len(categorical_vars) == 2:
            self.plot_interaction_2_vars(emm_df, categorical_vars, ms, out_dir)
        elif len(categorical_vars) == 3:
            self.plot_interaction_3_vars(emm_df, categorical_vars, ms, out_dir)
        else:
            print("This function supports only 2 or 3 categorical variables.")

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

        # # Create a third legend to link City to its color
        # city_legend_labels = [f'{city} (color)' for city in emm_df[categorical_vars[1]].unique()]
        # city_legend_handles = [plt.Line2D([0], [0], color=city_color_mapping[city]) for city in emm_df[categorical_vars[1]].unique()]
        # ax.legend(city_legend_handles, city_legend_labels, title='City-Color', loc='lower left')
        # ax2 = plt.gca().add_artist(ax.legend(age_legend_handles, age_legend_labels, title='Age_Group', loc='upper right'))

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
