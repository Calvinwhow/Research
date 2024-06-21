import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, kendalltau

class ScatterplotGenerator:
    def __init__(self, dataframe, data_dict, x_label='xlabel', y_label='ylabel', correlation='pearson', palette='tab10', out_dir=None, rows_per_fig=None, cols_per_fig=None, ylim=None, category_col=None, kde=False, swap_x_and_y=False):
        """
        Initialize the ScatterplotGenerator class.

        Parameters:
        - dataframe (pd.DataFrame): The dataframe containing the data.
        - data_dict (dict): A dictionary where keys are dependent variables and values are lists of independent variables.
        - x_label (str): Label for the x-axis.
        - y_label (str): Label for the y-axis.
        - correlation (str): Type of correlation ('pearson', 'spearman', or 'kendall').
        - palette (str): Color palette for the plots.
        - out_dir (str): Directory to save the plots.
        - rows_per_fig (int): Number of rows per figure.
        - cols_per_fig (int): Number of columns per figure.
        - ylim (tuple): Limits for the y-axis.
        - category_col (str): Column name for the category to color the scatter plot.
        - kde (bool): Whether to add KDE plots in the margins.
        - swap_x_and_y (bool): Whether to swap X and Y on the plot axes. 
        """
        self.dataframe = dataframe
        self.data_dict = data_dict
        self.x_label = x_label
        self.y_label = y_label
        self.correlation = correlation
        self.out_dir = out_dir
        self.palette = palette
        self.rows_per_fig = rows_per_fig or len(data_dict)
        self.cols_per_fig = cols_per_fig or len(data_dict)
        self.figures = []
        self.category_col = category_col
        self.ylim = ylim
        self.kde = kde
        self.swap_x_and_y = swap_x_and_y

    def set_palette(self):
        """Set the color palette for the plots."""
        sns.set_style('white')
        if isinstance(self.palette, str) and self.palette.startswith('#'):
            sns.set_palette(sns.color_palette([self.palette]))
        else:
            sns.set_palette(self.palette)

    def initialize_figure(self):
        """Initialize a new figure with subplots for each independent variable."""
        fig, axes = plt.subplots(self.rows_per_fig, self.cols_per_fig, figsize=(self.cols_per_fig * 5, self.rows_per_fig * 5))
        axes = axes.flatten()
        self.figures.append(fig)
        return fig, axes

    def plot_scatter_with_kde(self):
        """Create scatter plots with optional KDE for the given data."""
        for y_var, x_vars in self.data_dict.items():
            fig, axes = self.initialize_figure()
            current_ax = 0
            for x_var in x_vars:
                self.x_var = x_var; self.y_var = y_var
                x_var, y_var = self.swap_vars(x_var, y_var)
                if self.category_col:
                    self.plot_colored_scatter(axes[current_ax], x_var, y_var)
                else:
                    self.plot_single_scatter(axes[current_ax], x_var, y_var)

                if self.kde:
                    self.add_marginal_kde(axes[current_ax], x_var, y_var)
                
                current_ax += 1
                
    def swap_vars(self, x_var, y_var):
        """Swaps X and Y vars just for plotting"""
        if self.swap_x_and_y:
            int_var = x_var
            x_var = y_var
            y_var = int_var
        return x_var, y_var
        
    def plot_colored_scatter(self, ax, x_var, y_var, swap_vars=False):
        """
        Plot a scatter plot with regression and coloring by category.

        Parameters:
        - ax (matplotlib.axes.Axes): The axes to plot on.
        - x_var (str): The x variable for the plot.
        - y_var (str): The y variable for the plot.
        """

        sns.regplot(x=x_var, y=y_var, data=self.dataframe, ax=ax, scatter=False)
        sns.scatterplot(x=x_var, y=y_var, hue=self.category_col, data=self.dataframe, ax=ax, palette=self.palette)
        self.annotate_plot(ax, x_var, y_var)

    def plot_single_scatter(self, ax, independent_var, dependent_var):
        """
        Plot a single scatter plot with regression.

        Parameters:
        - ax (matplotlib.axes.Axes): The axes to plot on.
        - independent_var (str): The independent variable for the plot.
        - dependent_var (str): The dependent variable for the plot.
        """
        sns.regplot(x=independent_var, y=dependent_var, data=self.dataframe, ax=ax)
        self.annotate_plot(ax, independent_var, dependent_var)

    def annotate_plot(self, ax, independent_var, dependent_var):
        """
        Annotate the plot with correlation statistics and axis labels.

        Parameters:
        - ax (matplotlib.axes.Axes): The axes to annotate.
        - independent_var (str): The independent variable for the plot.
        - dependent_var (str): The dependent variable for the plot.
        """
        if self.correlation == 'pearson':
            r, p = pearsonr(self.dataframe[independent_var], self.dataframe[dependent_var])
        elif self.correlation == 'spearman':
            r, p = spearmanr(self.dataframe[independent_var], self.dataframe[dependent_var])
        elif self.correlation == 'kendall':
            r, p = kendalltau(self.dataframe[independent_var], self.dataframe[dependent_var])
        else:
            raise ValueError(f'Correlation {self.correlation} not specified, please select "pearson", "kendall" or "spearman"')

        if self.ylim is not None:
            ax.set_ylim(self.ylim[0], self.ylim[1])
        
        ax.set_title(self.x_var)
        ax.annotate(f"{self.correlation.capitalize()} r = {r:.2f}, p = {p:.3f}", xy=(.05, .95), xycoords='axes fraction', horizontalalignment='left', verticalalignment='top')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)

    def add_marginal_kde(self, ax, independent_var, dependent_var):
        """
        Add marginal KDE plots to the axes.

        Parameters:
        - ax (matplotlib.axes.Axes): The axes to add KDE plots to.
        - independent_var (str): The independent variable for the KDE plot.
        - dependent_var (str): The dependent variable for the KDE plot.
        """
        sns.kdeplot(data=self.dataframe, x=independent_var, hue=self.category_col, multiple="stack", palette=self.palette, ax=ax.twiny())
        sns.kdeplot(data=self.dataframe, y=dependent_var, hue=self.category_col, multiple="stack", palette=self.palette, ax=ax.twinx(), vertical=True)

    def save_plots(self):
        """Save the generated plots to the specified directory."""
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            for i, fig in enumerate(self.figures):
                fig.savefig(os.path.join(self.out_dir, f'{self.y_var}{i}.svg'))
                print(f'Figure saved to: {os.path.join(self.out_dir, f"{self.y_var}{i}")}')
                plt.show()

    def run(self):
        """Run the full pipeline: set palette, plot scatter plots, add marginal KDEs, and save plots."""
        self.set_palette()
        self.plot_scatter_with_kde()
        self.save_plots()
