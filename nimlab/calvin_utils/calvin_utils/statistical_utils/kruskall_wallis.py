from scipy import stats
import pandas as pd

class KruskalWallisTest:
    """
    A class to perform the Kruskal-Wallis H-test for independent samples.

    The Kruskal-Wallis H-test is a non-parametric method for testing whether samples 
    originate from the same distribution. It is used to compare two or more independent 
    groups of sampled data. This test is an extension of the Mann-Whitney U test to 
    multiple groups.

    Attributes
    ----------
    data : pd.DataFrame
        The data on which to perform the Kruskal-Wallis test.
    value_column : str
        The name of the column containing the values to test.
    group_column : str
        The name of the column containing the groups.

    Methods
    -------
    __init__(data, value_column, group_column)
        Initializes the Kruskal-WallisTest with the given data and column names.
    run_test()
        Performs the Kruskal-Wallis H-test and returns the results.

    When to Use
    -----------
    The Kruskal-Wallis test should be used when you want to test the null hypothesis 
    that the distributions of the groups are the same. It is particularly useful when 
    the assumptions of the one-way ANOVA are not met, such as when the data are not 
    normally distributed or when there are outliers that might affect the results of 
    ANOVA. The test requires the following conditions:
      - Independent samples from each group
      - Ordinal or continuous dependent variable
      - Independent variable with two or more levels (groups)

    Example usage
    -------------
    data = pd.DataFrame({'Value': [1, 2, 3, 4, 5, 6], 'Group': ['A', 'A', 'B', 'B', 'C', 'C']})
    kruskal_test = KruskalWallisTest(data, 'Value', 'Group')
    test_result = kruskal_test.run_test()
    print(test_result)
    """

    def __init__(self, data, value_column, group_column):
        """
        Initializes the Kruskal-WallisTest with the given data and column names.

        Parameters
        ----------
        data : pd.DataFrame
            The data on which to perform the Kruskal-Wallis test.
        value_column : str
            The name of the column containing the values to test.
        group_column : str
            The name of the column containing the groups.
        """
        self.data = data
        self.value_column = value_column
        self.group_column = group_column
        result = self.run_test()
        self.pretty_print_result(result)

    def run_test(self):
        """
        Performs the Kruskal-Wallis H-test on the provided data.

        Returns
        -------
        result : dict
            A dictionary containing the test statistic and the p-value.
        """
        # Grouping data by 'group_column' and applying the test
        grouped_data = [group[self.value_column].values for name, group in self.data.groupby(self.group_column)]
        kruskal_test_result = stats.kruskal(*grouped_data)

        # Output the test statistic and the p-value
        result = {
            "Kruskal-Wallis Test statistic": kruskal_test_result.statistic,
            "P-value": kruskal_test_result.pvalue
        }

        # Assess the p-value to determine the outcome of the hypothesis test
        if kruskal_test_result.pvalue < 0.05:
            result["Conclusion"] = "There is a statistically significant difference between groups."
        else:
            result["Conclusion"] = "There is no statistically significant difference between groups."

        return result
    
    def pretty_print_result(self, result):
        """
        Pretty prints the result of the Kruskal-Wallis H-test.

        Parameters
        ----------
        result : dict
            The result dictionary containing the test statistic, p-value, and conclusion.
        """
        print(f"Kruskal-Wallis Test Statistic: {result['Kruskal-Wallis Test statistic']:.4f}")
        print(f"P-value: {result['P-value']:.4f}")
        print(f"Conclusion: {result['Conclusion']}")